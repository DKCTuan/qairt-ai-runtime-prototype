#include "rf_model.h"
#include <stdio.h>

/* const ở cả mảng dữ liệu (rf_model.h) lẫn bảng con trỏ này: rừng nằm
   ở .rodata (flash) thay vì .data (RAM). */
const int   * const tree_features[NUM_TREES] = {
    tree_0_feature, tree_1_feature, tree_2_feature, tree_3_feature, tree_4_feature,
    tree_5_feature, tree_6_feature, tree_7_feature, tree_8_feature, tree_9_feature,
    tree_10_feature, tree_11_feature, tree_12_feature, tree_13_feature, tree_14_feature,
    tree_15_feature, tree_16_feature, tree_17_feature, tree_18_feature, tree_19_feature};

const float * const tree_thresholds[NUM_TREES] = {
    tree_0_threshold, tree_1_threshold, tree_2_threshold, tree_3_threshold, tree_4_threshold,
    tree_5_threshold, tree_6_threshold, tree_7_threshold, tree_8_threshold, tree_9_threshold,
    tree_10_threshold, tree_11_threshold, tree_12_threshold, tree_13_threshold, tree_14_threshold,
    tree_15_threshold, tree_16_threshold, tree_17_threshold, tree_18_threshold, tree_19_threshold};

const int   * const tree_lefts[NUM_TREES] = {
    tree_0_left, tree_1_left, tree_2_left, tree_3_left, tree_4_left,
    tree_5_left, tree_6_left, tree_7_left, tree_8_left, tree_9_left,
    tree_10_left, tree_11_left, tree_12_left, tree_13_left, tree_14_left,
    tree_15_left, tree_16_left, tree_17_left, tree_18_left, tree_19_left};

const int   * const tree_rights[NUM_TREES] = {
    tree_0_right, tree_1_right, tree_2_right, tree_3_right, tree_4_right,
    tree_5_right, tree_6_right, tree_7_right, tree_8_right, tree_9_right,
    tree_10_right, tree_11_right, tree_12_right, tree_13_right, tree_14_right,
    tree_15_right, tree_16_right, tree_17_right, tree_18_right, tree_19_right};

const float (* const tree_values[NUM_TREES])[NUM_CLASSES] = {
    tree_0_value, tree_1_value, tree_2_value, tree_3_value, tree_4_value,
    tree_5_value, tree_6_value, tree_7_value, tree_8_value, tree_9_value,
    tree_10_value, tree_11_value, tree_12_value, tree_13_value, tree_14_value,
    tree_15_value, tree_16_value, tree_17_value, tree_18_value, tree_19_value};

int predict_forest(float input[], float output_probs[]) {
  int t, c;
  for (c = 0; c < NUM_CLASSES; c++) output_probs[c] = 0.0f;

  for (t = 0; t < NUM_TREES; t++) {
    int node = 0;
    while (tree_lefts[t][node] != -1) {
      int fi = tree_features[t][node];
      if (input[fi] <= tree_thresholds[t][node]) node = tree_lefts[t][node];
      else                                       node = tree_rights[t][node];
    }
    for (c = 0; c < NUM_CLASSES; c++) output_probs[c] += tree_values[t][node][c];
  }

  float total = 0.0f;
  for (c = 0; c < NUM_CLASSES; c++) total += output_probs[c];
  if (total > 0.0f) for (c = 0; c < NUM_CLASSES; c++) output_probs[c] /= total;

  int best = 0; float mx = -1.0f;
  for (c = 0; c < NUM_CLASSES; c++)
      if (output_probs[c] > mx) { mx = output_probs[c]; best = c; }
  return best;
}

#ifdef RF_MODEL_TEST
int main(void) {
    const char* class_names[NUM_CLASSES] = {"Background", "Game", "RTVideo", "VStream", "Voice"};
    /* một window Background thật (gần trung vị của lớp) */
    float input[RF_N_FEATURES] = {0, 26.9784, 832.711, 100, 60, 19.7368, 0, 0, 0.000524, 0.995556, 0};
    float probs[NUM_CLASSES];
    int cls = predict_forest(input, probs);
    printf("Predicted: %d (%s)\n", cls, class_names[cls]);
    for (int i = 0; i < NUM_CLASSES; i++) printf("  %-11s %.4f\n", class_names[i], probs[i]);
    return 0;
}
#endif