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
#include "slimt/QMM.hh"
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

// In-place variant of select_batch for tensors the decode loop owns: `keep`
// is ascending, so every kept slab moves to an offset at or before its
// source within the same buffer. No fresh allocation means the pages stay
// resident across compactions (a fresh heap tensor per finished sentence
// showed up as ~4% of runtime in page-fault handling).
void compact_batch(Tensor &tensor, const std::vector<size_t> &keep) {
  size_t batch_size = tensor.dim(0);
  size_t bytes_per_entry =
      size_in_bytes(tensor.type()) * tensor.size() / batch_size;
  char *data = tensor.data<char>();
  for (size_t i = 0; i < keep.size(); ++i) {
    if (keep[i] == i) {
      continue;
    }
    std::memmove(data + i * bytes_per_entry, data + keep[i] * bytes_per_entry,
                 bytes_per_entry);
  }
  tensor.shape().set_dim(0, keep.size());
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

Histories Model::decode(const Tensor &encoder_out, const Input &input,
                        Arena &arena) const {
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  const Decoder &decoder = transformer_.decoder();

  // One shortlisted output projection per distinct request in the batch
  // (rows of one request share their shortlist's shared_ptr, so grouping is
  // pointer equality). The column-select on `W` and the bias gather run once
  // per decode; each step then multiplies a request's rows only against its
  // own ~hundreds of candidates instead of a batch-wide union. A null entry
  // is the full-vocabulary group, for rows whose request has no shortlist.
  // Built before any ArenaScope so the tensors heap-allocate and survive
  // across step iterations.
  const RowShortlists &row_shortlists = input.shortlist_rows();
  std::vector<std::shared_ptr<const Words>> group_words;
  std::vector<SelectedAffine> group_selected;
  std::vector<size_t> row_group(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    std::shared_ptr<const Words> words =
        row_shortlists.empty() ? nullptr : row_shortlists[i];
    size_t group = 0;
    while (group < group_words.size() && group_words[group] != words) {
      group++;
    }
    if (group == group_words.size()) {
      group_words.push_back(words);
      group_selected.push_back(words ? decoder.prepare_shortlisted_output(*words)
                                     : SelectedAffine{});
    }
    row_group[i] = group;
  }

  std::vector<bool> complete(batch_size, false);
  const Vocabulary &target_vocab = target_vocabulary();
  uint32_t eos = target_vocab.eos_id();
  // Cap each sentence's target length from its own (unpadded) source length.
  // A cap derived from the batch's padded length would let a sentence run
  // longer the longer its co-batched neighbours are, making the output
  // depend on batch composition. The additive slack exists because a purely
  // multiplicative cap starves short sources — "CHAPTER 102. A Bower in the
  // Arsacides." is 16 subwords, and 1.5×16 = 24 truncated the Spanish
  // mid-word ("...los Arsaci") — while barely moving the runaway bound for
  // long ones.
  constexpr size_t kTargetLengthSlack = 8;
  std::vector<size_t> target_caps(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    target_caps[i] =
        input.limit_factor() * input.lengths()[i] + kTargetLengthSlack;
  }
  auto record = [eos, &complete, &target_caps](
                    const std::vector<size_t> &active_to_original, Words &step,
                    Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      size_t original_id = active_to_original[i];
      if (not complete[original_id]) {
        sentences[original_id].push_back(step[i]);
        complete[original_id] =
            (step[i] == eos) ||
            sentences[original_id].size() >= target_caps[original_id];
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
  // intermediates, returned logits/attn) come from the shared scratch arena;
  // allocations that must outlive the step (states, contexts, encoder_out,
  // select_batch outputs from compact) happen outside arena scopes. The arena
  // arrives holding the now-dead encoder transients (encoder_out was cloned to
  // the heap), so reset before the first step reclaims that space.
  arena.reset();
  // Loop bound only; must dominate every per-row cap (lengths[i] ≤ the
  // padded source_sequence_length). Rows stop at their own cap via record().
  size_t max_seq_length =
      input.limit_factor() * source_sequence_length + kTargetLengthSlack;

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

    if (active_encoder_out != &selected_encoder_out) {
      // First shrink: encoder_out and mask are caller-owned consts, so the
      // initial compaction copies them once; every later one is in place.
      selected_encoder_out =
          select_batch(*active_encoder_out, keep, active_encoder_out->name());
      selected_mask = select_batch(*active_mask, keep, active_mask->name());
      active_encoder_out = &selected_encoder_out;
      active_mask = &selected_mask;
    } else {
      compact_batch(selected_encoder_out, keep);
      compact_batch(selected_mask, keep);
    }
    for (Tensor &state : states) {
      compact_batch(state, keep);
    }
    for (AttentionContext &context : contexts) {
      compact_batch(context.keys, keep);
      compact_batch(context.values, keep);
    }
    active_to_original = std::move(next_active_to_original);
  };

  // Project the step's hidden states group by group and sample. Each
  // group's rows are gathered into a contiguous slab (skipped when one
  // group covers every active row, the common single-request case); the
  // gathered slab and logits are step transients, so this must run inside
  // the step's ArenaScope.
  auto sample_step = [&](const Tensor &hidden) {
    size_t active_rows = active_to_original.size();
    size_t hidden_dim = hidden.dim(-1);
    Words sampled(active_rows);

    std::vector<std::vector<size_t>> members(group_words.size());
    for (size_t i = 0; i < active_rows; ++i) {
      members[row_group[active_to_original[i]]].push_back(i);
    }

    const float *hidden_data = hidden.data<float>();
    for (size_t g = 0; g < members.size(); ++g) {
      if (members[g].empty()) {
        continue;
      }
      Tensor gathered;
      const Tensor *x = &hidden;
      if (members[g].size() != active_rows) {
        gathered = Tensor(Type::f32, Shape({members[g].size(), 1, hidden_dim}),
                          "grouped_hidden");
        float *target = gathered.data<float>();
        for (size_t j = 0; j < members[g].size(); ++j) {
          const float *source = hidden_data + members[g][j] * hidden_dim;
          std::copy(source, source + hidden_dim, target + j * hidden_dim);
        }
        x = &gathered;
      }

      Words group_sample;
      if (group_words[g]) {
        Tensor logits = affine_with_selected(group_selected[g], *x, "logits");
        group_sample = greedy_sample_from_words(logits, target_vocab,
                                                *group_words[g],
                                                members[g].size());
      } else {
        Tensor logits = decoder.project(*x);
        group_sample = greedy_sample(logits, target_vocab, members[g].size());
      }
      for (size_t j = 0; j < members[g].size(); ++j) {
        sampled[members[g][j]] = group_sample[j];
      }
    }
    return sampled;
  };

  size_t remaining;
  {
    ArenaScope arena_scope(arena);
    auto [hidden, attn] = decoder.step(*active_encoder_out, *active_mask,
                                       states, contexts, previous_slice,
                                       /*step_index=*/0);
    previous_slice = sample_step(hidden);
    update_alignment(active_to_original, input.lengths(), complete, attn,
                     alignments);
    remaining = record(active_to_original, previous_slice, sentences);
  }
  compact();

  size_t steps = 1;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    arena.reset();
    decoder_rows += active_to_original.size();
    {
      ArenaScope arena_scope(arena);
      auto [hidden, attn] = decoder.step(*active_encoder_out, *active_mask,
                                         states, contexts, previous_slice,
                                         /*step_index=*/i);
      steps++;
      previous_slice = sample_step(hidden);
      update_alignment(active_to_original, input.lengths(), complete, attn,
                       alignments);
      remaining = record(active_to_original, previous_slice, sentences);
    }
    compact();
  }

  // The per-step output projections cached ruy packs of the per-request
  // shortlisted Ws (heap-owned, pointer-keyed cache); drop them before
  // `group_selected` is freed so a later allocation reusing an address
  // can't hit a stale pack.
  qmm::clear_standalone_pack_cache();

  if (std::getenv("SLIMT_DECODE_STATS") != nullptr) {
    size_t target_tokens = 0;
    for (const auto &sentence : sentences) {
      target_tokens += sentence.size();
    }
    size_t wasted_rows =
        decoder_rows > target_tokens ? decoder_rows - target_tokens : 0;
    size_t shortlisted_groups = 0;
    for (const auto &words : group_words) {
      shortlisted_groups += words ? 1 : 0;
    }
    std::fprintf(stderr,
                 "[decode-stats] batch=%zu src_len=%zu steps=%zu rows=%zu "
                 "target_tokens=%zu wasted_rows=%zu limit=%zu groups=%zu "
                 "shortlisted_groups=%zu\n",
                 batch_size, source_sequence_length, steps, decoder_rows,
                 target_tokens, wasted_rows, max_seq_length,
                 group_words.size(), shortlisted_groups);
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

  // One scratch arena per worker thread, shared by the encoder and the decode
  // loop. Encoder transients (word embedding, per-layer Q/K/V/O and FFN
  // projections, self-attention scores) come from it; encoder_out is cloned
  // out to the heap once the scope closes so it survives into decode(), which
  // then resets and reuses the same arena per step.
  constexpr size_t kArenaInitialBytes = 8 << 20;  // 8 MiB
  static thread_local Arena arena(kArenaInitialBytes);
  arena.reset();
  Tensor encoder_out;
  {
    Tensor arena_encoder_out;
    {
      ArenaScope arena_scope(arena);
      Tensor word_embedding =
          index_select(transformer_.embedding(), indices, "word_embedding");
      transform_embedding(word_embedding);

      // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
      // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
      arena_encoder_out = transformer_.encoder().forward(word_embedding, mask);
    }
    encoder_out = arena_encoder_out.clone("encoder_out");
  }

  Histories histories = decode(encoder_out, input, arena);
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
