CC ?= gcc
STRIP ?= strip
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I. -Igenerated -I../traffic_models
OUT ?= ../dist/traffic_models_p16
GRU_LIBRARY ?=
GRU_INCLUDE ?= ../tflite_qnn_prototype/include

ifneq ($(GRU_LIBRARY),)
GRU_FLAGS = -DTRAFFIC_P16_WITH_GRU -I$(GRU_INCLUDE)
GRU_LINK = -L$(dir $(GRU_LIBRARY)) -l:$(notdir $(GRU_LIBRARY)) -Wl,-rpath,'$$ORIGIN'
APP_GRU_LINK = -Wl,-rpath-link,$(dir $(GRU_LIBRARY))
endif

.PHONY: all combined test
all: $(OUT)/libtraffic_p16.so $(OUT)/app_p16_arm64

# A single public SDK for the current production pair: Random Forest + the
# two-input TinyGRU P16.  The old one-input GRU is intentionally not linked.
# It keeps the deployment to libtraffic_ai.so + its one P16 model dependency.
combined: $(OUT)/libtraffic_ai.so $(OUT)/app_traffic_ai_arm64

$(OUT):
	mkdir -p $(OUT)

$(OUT)/libtraffic_p16.so: traffic_p16.c traffic_p16.h \
	generated/traffic_p16_deployment_config.c generated/traffic_p16_deployment_config.h \
	exports.map Makefile $(GRU_LIBRARY) | $(OUT)
	$(CC) $(CFLAGS) $(GRU_FLAGS) -fPIC -shared traffic_p16.c \
		generated/traffic_p16_deployment_config.c $(GRU_LINK) -lm \
		-Wl,--gc-sections -Wl,--version-script=exports.map \
		-Wl,-soname,libtraffic_p16.so -o $@
	$(STRIP) --strip-unneeded $@

$(OUT)/app_p16_arm64: examples/p16_smoke.c traffic_p16.h generated/traffic_p16_deployment_config.h Makefile $(OUT)/libtraffic_p16.so
	$(CC) $(CFLAGS) examples/p16_smoke.c -L$(OUT) -ltraffic_p16 $(APP_GRU_LINK) \
		-Wl,-rpath,'$$ORIGIN:$$ORIGIN/../lib' -o $@
	$(STRIP) --strip-unneeded $@

$(OUT)/libtraffic_ai.so: traffic_p16.c traffic_p16.h \
	generated/traffic_p16_deployment_config.c generated/traffic_p16_deployment_config.h \
	../traffic_models/traffic_models.c ../traffic_models/traffic_models.h \
	../traffic_models/rf_feature_extractor.c ../traffic_models/vendor/rf_model.c \
	exports_combined.map Makefile $(GRU_LIBRARY) | $(OUT)
	$(CC) $(CFLAGS) $(GRU_FLAGS) -fPIC -shared \
		../traffic_models/traffic_models.c ../traffic_models/rf_feature_extractor.c \
		../traffic_models/vendor/rf_model.c traffic_p16.c \
		generated/traffic_p16_deployment_config.c $(GRU_LINK) -lm \
		-Wl,--gc-sections -Wl,--version-script=exports_combined.map \
		-Wl,-soname,libtraffic_ai.so -o $@
	$(STRIP) --strip-unneeded $@

$(OUT)/app_traffic_ai_arm64: examples/p16_smoke.c traffic_p16.h generated/traffic_p16_deployment_config.h Makefile $(OUT)/libtraffic_ai.so
	$(CC) $(CFLAGS) examples/p16_smoke.c -L$(OUT) -ltraffic_ai $(APP_GRU_LINK) \
		-Wl,-rpath,'$$ORIGIN:$$ORIGIN/../lib' -o $@
	$(STRIP) --strip-unneeded $@

test:
	$(CC) $(CFLAGS) traffic_p16.c generated/traffic_p16_deployment_config.c tests/test_preprocessing.c -lm -o /tmp/traffic_p16_test
	/tmp/traffic_p16_test
