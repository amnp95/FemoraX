#include "solvers/csr_operator.hpp"

namespace femorax::solvers {

namespace {

template <class T>
Kokkos::View<T*, Space> upload(const std::vector<T>& h, const char* label) {
  Kokkos::View<T*, Space> d(std::string(label), h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

}  // namespace

CsrOperator::CsrOperator(const std::vector<std::int64_t>& row_offsets, const std::vector<std::int32_t>& columns,
                         Symmetry symmetry, std::vector<Field> fields) {
  info_.rows = row_offsets.empty() ? 0 : static_cast<std::int64_t>(row_offsets.size()) - 1;
  info_.symmetry = symmetry;
  info_.fields = fields.empty() ? std::vector<Field>{{"all", 0, info_.rows}} : std::move(fields);
  row_offsets_ = upload(row_offsets, "csr_row_offsets");
  columns_ = upload(columns, "csr_columns");
  values_ = Kokkos::View<double*, Space>("csr_values", columns.size());
  matrix_ = {row_offsets_, columns_, values_};
}

void CsrOperator::apply(const ConstVector& x, const Vector& y) const {
  const auto ro = row_offsets_;
  const auto col = columns_;
  const auto val = values_;
  Kokkos::parallel_for(
      "csr_spmv", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, info_.rows),
      KOKKOS_LAMBDA(const std::int64_t i) {
        double s = 0.0;
        for (std::int64_t k = ro(i); k < ro(i + 1); ++k) s += val(k) * x(col(k));
        y(i) = s;
      });
}

void CsrOperator::diagonal(const Vector& d) const {
  const auto ro = row_offsets_;
  const auto col = columns_;
  const auto val = values_;
  Kokkos::parallel_for(
      "csr_diagonal", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, info_.rows),
      KOKKOS_LAMBDA(const std::int64_t i) {
        double v = 0.0;
        for (std::int64_t k = ro(i); k < ro(i + 1); ++k)
          if (col(k) == i) v = val(k);
        d(i) = v;
      });
}

}  // namespace femorax::solvers
