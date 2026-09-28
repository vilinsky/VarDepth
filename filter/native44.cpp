#include <cmath>
#include <cstdint>

// No fast-math: preserve float64 comparisons, missing-value routing, and
// the original sequential order of additions for each row.
extern "C" void logits44(
    const double* x, int64_t rows, int64_t features,
    const int32_t* roots, int64_t trees,
    const int32_t* feature, const int32_t* left, const int32_t* right,
    const int32_t* missing_left, const double* threshold,
    const double* value, double baseline, double* out) {
  for (int64_t r = 0; r < rows; ++r) {
    double score = baseline;
    for (int64_t t = 0; t < trees; ++t) {
      int32_t node = roots[t];
      while (feature[node] >= 0) {
        double v = x[r * features + feature[node]];
        bool go_left = std::isnan(v) ? missing_left[node] != 0 : v <= threshold[node];
        node = go_left ? left[node] : right[node];
      }
      score += value[node];
    }
    out[r] = score;
  }
}
