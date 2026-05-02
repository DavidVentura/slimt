#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "slimt/Tensor.hh"
#include "slimt/Types.hh"

namespace slimt {

class Input {
 public:
  Input(size_t batch_size, size_t sequence_length, uint32_t pad_id,
        float limit_factor);

  void add(const std::vector<uint32_t> &words);
  void finalize();

  const Tensor &indices() const { return batch_; }
  const Tensor &mask() const { return mask_; }
  const std::vector<uint32_t> &words() const { return words_; }
  void set_shortlist_words(std::shared_ptr<const Words> words) {
    shortlist_words_ = std::move(words);
  }
  const std::shared_ptr<const Words> &shortlist_words() const {
    return shortlist_words_;
  }
  const std::vector<size_t> &lengths() const { return lengths_; }
  size_t index() const { return index_; }
  float occupancy();
  float limit_factor() const;

 private:
  std::vector<uint32_t> words_;
  std::shared_ptr<const Words> shortlist_words_;
  std::vector<size_t> lengths_;
  Tensor batch_;
  Tensor mask_;
  size_t index_ = 0;
  uint32_t pad_id_ = 0;
  size_t used_ = 0;
  float limit_factor_;
  bool finalized_ = false;
};
}  // namespace slimt
