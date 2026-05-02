#include "slimt/Model.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Arena.hh"
#include "slimt/Input.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

size_t model_id = 0;

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  auto maybe_mmap = [](const std::string &path) {
    return path.empty() ? io::MmapFile() : io::MmapFile(path);
  };

  return {
      .model = maybe_mmap(package.model),                          //
      .vocabulary = maybe_mmap(package.vocabulary),                //
      .target_vocabulary = maybe_mmap(package.target_vocabulary),  //
      .shortlist = maybe_mmap(package.shortlist),                  //
      .ssplit = maybe_mmap(package.ssplit),                        //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .target_vocabulary = {mmap.target_vocabulary.data(),             //
                            mmap.target_vocabulary.size()},            //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
      .ssplit = {mmap.ssplit.data(), mmap.ssplit.size()},              //
  };
}

std::unique_ptr<Vocabulary> maybe_target_vocabulary(View target_view) {
  if (target_view.data == nullptr || target_view.size == 0) {
    return nullptr;
  }
  return std::make_unique<Vocabulary>(target_view);
}

}  // namespace

Model::Model(const Config &config, const Package<View> &package)
    : id_(model_id++),
      config_(config),
      view_(package),
      src_vocabulary_(package.vocabulary),
      tgt_vocabulary_(maybe_target_vocabulary(package.target_vocabulary)),
      processor_(src_vocabulary_),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, package.model),
      shortlist_generator_(make_shortlist_generator(
          package.shortlist, src_vocabulary_, target_vocabulary())) {}

Model::Model(const Config &config, const Package<std::string> &package)
    : id_(model_id++),
      config_(config),
      mmap_(mmap_from(package)),
      view_(view_from(*mmap_)),
      src_vocabulary_(view_.vocabulary),
      tgt_vocabulary_(maybe_target_vocabulary(view_.target_vocabulary)),
      processor_(src_vocabulary_),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, view_.model),
      shortlist_generator_(make_shortlist_generator(
          view_.shortlist, src_vocabulary_, target_vocabulary())) {}

std::optional<ShortlistGenerator> Model::make_shortlist_generator(
    View view, const Vocabulary &source, const Vocabulary &target) {
  if (view.data == nullptr || view.size == 0) {
    return std::nullopt;
  }
  return ShortlistGenerator(view, source, target);
}

namespace {
Tensor select_batch(const Tensor &tensor, const std::vector<size_t> &indices,
                    const std::string &name) {
  Shape shape = tensor.shape();
  shape.set_dim(0, indices.size());
  Tensor selected(tensor.type(), shape, name);

  if (indices.empty()) {
    return selected;
  }

  size_t batch_size = tensor.dim(0);
  size_t bytes_per_entry =
      size_in_bytes(tensor.type()) * tensor.size() / batch_size;
  const char *source = tensor.data<char>();
  char *target = selected.data<char>();
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t index = indices[i];
    std::memcpy(target + i * bytes_per_entry, source + index * bytes_per_entry,
                bytes_per_entry);
  }

  return selected;
}

std::vector<Tensor> select_batch(const std::vector<Tensor> &tensors,
                                 const std::vector<size_t> &indices) {
  std::vector<Tensor> selected;
  selected.reserve(tensors.size());
  for (const Tensor &tensor : tensors) {
    selected.push_back(select_batch(tensor, indices, tensor.name()));
  }
  return selected;
}

AttentionContext select_batch(const AttentionContext &context,
                              const std::vector<size_t> &indices) {
  return {
      .keys = select_batch(context.keys, indices, context.keys.name()),
      .values = select_batch(context.values, indices, context.values.name()),
  };
}

std::vector<AttentionContext> select_batch(
    const std::vector<AttentionContext> &contexts,
    const std::vector<size_t> &indices) {
  std::vector<AttentionContext> selected;
  selected.reserve(contexts.size());
  for (const AttentionContext &context : contexts) {
    selected.push_back(select_batch(context, indices));
  }
  return selected;
}

void update_alignment(const std::vector<size_t> &active_to_original,
                      const std::vector<size_t> &lengths,
                      const std::vector<bool> &finished, const Tensor &attn,
                      Alignments &alignments) {
  const auto *data = attn.data<float>();
  size_t batch_size = attn.dim(-4);
  size_t num_heads = attn.dim(-3);
  size_t slice = attn.dim(-2);
  size_t source_length = attn.dim(-1);

  for (size_t id = 0; id < batch_size; id++) {
    size_t original_id = active_to_original[id];
    if (!finished[original_id]) {
      size_t batch_stride = (num_heads * slice * source_length);
      size_t head_stride = (slice * source_length);
      const float *alignment = data + id * batch_stride + head_stride * 0;
      size_t length = lengths[original_id];
      Distribution distribution(length);
      std::copy(alignment, alignment + length, distribution.data());
      alignments[original_id].push_back(std::move(distribution));
    }
  }
}
}  // namespace

Histories Model::decode(const Tensor &encoder_out, const Input &input) const {
  // Prepare a shortlist for the entire input.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  std::optional<Words> indices = std::nullopt;
  if (shortlist_generator_) {
    Shortlist shortlist = shortlist_generator_->generate(input.words());
    indices = shortlist.words();
  }
  // The following can be used to check if shortlist is going wrong.
  // std::vector<uint32_t> indices(target_vocab.size());
  // std::iota(indices.begin(), indices.end(), 0);

  std::vector<bool> complete(batch_size, false);
  const Vocabulary &target_vocab = target_vocabulary();
  uint32_t eos = target_vocab.eos_id();
  auto record = [eos, &complete](const std::vector<size_t> &active_to_original,
                                 Words &step, Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      size_t original_id = active_to_original[i];
      if (not complete[original_id]) {
        complete[original_id] = (step[i] == eos);
        sentences[original_id].push_back(step[i]);
      }
    }
    for (bool done : complete) {
      finished += static_cast<int>(done);
    }
    return sentences.size() - finished;
  };

  // Initialize a first step.
  Sentences sentences(batch_size);
  Alignments alignments(sentences.size());

  const Decoder &decoder = transformer_.decoder();
  std::vector<size_t> active_to_original(batch_size);
  std::iota(active_to_original.begin(), active_to_original.end(), 0);

  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  std::vector<AttentionContext> contexts = decoder.prepare_contexts(encoder_out);

  const Tensor *active_encoder_out = &encoder_out;
  const Tensor *active_mask = &input.mask();
  Tensor selected_encoder_out;
  Tensor selected_mask;

  size_t decoder_rows = active_to_original.size();

  // Per-step transient tensors (Q/K/V projections, attention scores, FFN
  // intermediates, returned logits/attn) come from this arena; allocations
  // that must outlive the step (states, contexts, encoder_out, select_batch
  // outputs from compact) happen outside arena scopes.
  constexpr size_t kArenaInitialBytes = 8 << 20;  // 8 MiB
  Arena arena(kArenaInitialBytes);

  size_t remaining;
  {
    ArenaScope arena_scope(arena);
    auto [logits, attn] = decoder.step(*active_encoder_out, *active_mask,
                                       states, contexts, previous_slice,
                                       indices, /*step_index=*/0);

    if (indices) {
      previous_slice = greedy_sample_from_words(logits, target_vocab, *indices,
                                                batch_size);
    } else {
      previous_slice = greedy_sample(logits, target_vocab, batch_size);
    }

    update_alignment(active_to_original, input.lengths(), complete, attn,
                     alignments);
    remaining = record(active_to_original, previous_slice, sentences);
  }

  auto compact = [&]() {
    std::vector<size_t> keep;
    std::vector<size_t> next_active_to_original;
    Words next_previous_slice;
    keep.reserve(active_to_original.size());
    next_active_to_original.reserve(active_to_original.size());
    next_previous_slice.reserve(active_to_original.size());

    for (size_t i = 0; i < active_to_original.size(); ++i) {
      size_t original_id = active_to_original[i];
      if (!complete[original_id]) {
        keep.push_back(i);
        next_active_to_original.push_back(original_id);
        next_previous_slice.push_back(previous_slice[i]);
      }
    }

    previous_slice = std::move(next_previous_slice);
    if (keep.empty()) {
      active_to_original.clear();
      return;
    }
    if (keep.size() == active_to_original.size()) {
      active_to_original = std::move(next_active_to_original);
      return;
    }

    selected_encoder_out =
        select_batch(*active_encoder_out, keep, active_encoder_out->name());
    selected_mask = select_batch(*active_mask, keep, active_mask->name());
    active_encoder_out = &selected_encoder_out;
    active_mask = &selected_mask;
    states = select_batch(states, keep);
    contexts = select_batch(contexts, keep);
    active_to_original = std::move(next_active_to_original);
  };
  compact();

  size_t max_seq_length = input.limit_factor() * source_sequence_length;
  size_t steps = 1;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    arena.reset();
    decoder_rows += active_to_original.size();
    {
      ArenaScope arena_scope(arena);
      auto [logits, attn] = decoder.step(*active_encoder_out, *active_mask,
                                         states, contexts, previous_slice,
                                         indices, /*step_index=*/i);
      steps++;
      if (indices) {
        previous_slice =
            greedy_sample_from_words(logits, target_vocab, *indices,
                                     active_to_original.size());
      } else {
        previous_slice =
            greedy_sample(logits, target_vocab, active_to_original.size());
      }
      update_alignment(active_to_original, input.lengths(), complete, attn,
                       alignments);
      remaining = record(active_to_original, previous_slice, sentences);
    }
    compact();
  }

  if (std::getenv("SLIMT_DECODE_STATS") != nullptr) {
    size_t target_tokens = 0;
    for (const auto &sentence : sentences) {
      target_tokens += sentence.size();
    }
    size_t wasted_rows =
        decoder_rows > target_tokens ? decoder_rows - target_tokens : 0;
    std::fprintf(stderr,
                 "[decode-stats] batch=%zu src_len=%zu steps=%zu rows=%zu "
                 "target_tokens=%zu wasted_rows=%zu limit=%zu shortlist=%d\n",
                 batch_size, source_sequence_length, steps, decoder_rows,
                 target_tokens, wasted_rows, max_seq_length,
                 indices.has_value() ? 1 : 0);
  }

  Histories histories;
  for (size_t i = 0; i < sentences.size(); i++) {
    Hypothesis hypothesis{
        .target = std::move(sentences[i]),     //
        .alignment = std::move(alignments[i])  //
    };
    auto history = std::make_shared<Hypothesis>(std::move(hypothesis));
    histories.push_back(std::move(history));
  }

  return histories;
}

Histories Model::forward(const Input &input) const {
  const Tensor &indices = input.indices();
  const Tensor &mask = input.mask();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  Tensor word_embedding =
      index_select(transformer_.embedding(), indices, "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  Tensor encoder_out = transformer_.encoder().forward(word_embedding, mask);
  Histories histories = decode(encoder_out, input);
  return histories;
}

namespace preset {
Model::Config tiny() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 6,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
  };
  // NOLINTEND
  return config;
}

Model::Config base() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 6,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
  };
  // NOLINTEND
  return config;
}

Model::Config nano() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 4,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
  };
  // NOLINTEND
  return config;
}
}  // namespace preset

}  // namespace slimt
