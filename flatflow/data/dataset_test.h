// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_DATASETTEST_H_
#define FLATBUFFERS_GENERATED_DATASETTEST_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 24 &&
              FLATBUFFERS_VERSION_MINOR == 3 &&
              FLATBUFFERS_VERSION_REVISION == 7,
             "Non-compatible flatbuffers version included");

struct Sizes;
struct SizesBuilder;

struct Sizes FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef SizesBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_SIZES = 4
  };
  const ::flatbuffers::Vector64<uint16_t> *sizes() const {
    return GetPointer64<const ::flatbuffers::Vector64<uint16_t> *>(VT_SIZES);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset64(verifier, VT_SIZES) &&
           verifier.VerifyVector(sizes()) &&
           verifier.EndTable();
  }
};

struct SizesBuilder {
  typedef Sizes Table;
  ::flatbuffers::FlatBufferBuilder64 &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_sizes(::flatbuffers::Offset64<::flatbuffers::Vector64<uint16_t>> sizes) {
    fbb_.AddOffset(Sizes::VT_SIZES, sizes);
  }
  explicit SizesBuilder(::flatbuffers::FlatBufferBuilder64 &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Sizes> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Sizes>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Sizes> CreateSizes(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    ::flatbuffers::Offset64<::flatbuffers::Vector64<uint16_t>> sizes = 0) {
  SizesBuilder builder_(_fbb);
  builder_.add_sizes(sizes);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<Sizes> CreateSizesDirect(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    const std::vector<uint16_t> *sizes = nullptr) {
  auto sizes__ = sizes ? _fbb.CreateVector64(*sizes) : 0;
  return CreateSizes(
      _fbb,
      sizes__);
}

inline const Sizes *GetSizes(const void *buf) {
  return ::flatbuffers::GetRoot<Sizes>(buf);
}

inline const Sizes *GetSizePrefixedSizes(const void *buf) {
  return ::flatbuffers::GetSizePrefixedRoot<Sizes,::flatbuffers::uoffset64_t>(buf);
}

inline bool VerifySizesBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<Sizes>(nullptr);
}

inline bool VerifySizePrefixedSizesBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<Sizes,::flatbuffers::uoffset64_t>(nullptr);
}

inline void FinishSizesBuffer(
    ::flatbuffers::FlatBufferBuilder64 &fbb,
    ::flatbuffers::Offset<Sizes> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedSizesBuffer(
    ::flatbuffers::FlatBufferBuilder64 &fbb,
    ::flatbuffers::Offset<Sizes> root) {
  fbb.FinishSizePrefixed(root);
}

#endif  // FLATBUFFERS_GENERATED_DATASETTEST_H_
