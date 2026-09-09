CC = gcc
STRIP = strip
OUT ?= ../dist/traffic_models
CFLAGS = -std=c11 -Os -Wall -Wextra -ffunction-sections -fdata-sections
# Optional Tiny GRU library, produced by model_deploy.py --shared.
GRU_LIBRARY ?=
GRU_INCLUDE ?= ../tflite_qnn_prototype/include
ifneq ($(GRU_LIBRARY),)
GRU_FLAGS = -DTRAFFIC_WITH_GRU -I$(GRU_INCLUDE)
GRU_LINK = -L$(dir $(GRU_LIBRARY)) -l:$(notdir $(GRU_LIBRARY)) -Wl,-rpath,'$$ORIGIN'
APP_GRU_LINK = -Wl,-rpath-link,$(dir $(GRU_LIBRARY))
endif
.PHONY: all FORCE
all: $(OUT)/app_arm64
FORCE:
$(OUT):
	mkdir -p $(OUT)
$(OUT)/libtraffic_models.so: traffic_models.c rf_feature_extractor.c traffic_models.h vendor/rf_model.c vendor/rf_model.h exports.map $(GRU_LIBRARY) FORCE | $(OUT)
	$(CC) $(CFLAGS) $(GRU_FLAGS) -fPIC -shared traffic_models.c rf_feature_extractor.c vendor/rf_model.c $(GRU_LINK) -lm -Wl,--gc-sections -Wl,--version-script=exports.map -Wl,-soname,libtraffic_models.so -o $@
	$(STRIP) --strip-unneeded $@
$(OUT)/app_arm64: app.c traffic_models.h $(OUT)/libtraffic_models.so
	$(CC) $(CFLAGS) app.c -I. -L$(OUT) -ltraffic_models $(APP_GRU_LINK) -Wl,-rpath,'$$ORIGIN' -o $@
	$(STRIP) --strip-unneeded $@
