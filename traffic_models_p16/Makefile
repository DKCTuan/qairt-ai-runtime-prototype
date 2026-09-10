CC ?= gcc
STRIP ?= strip
PYTHON ?= python3
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I. -I$(MODEL_PROFILE_DIR) -I../traffic_models
OUT ?= ../dist/traffic_models_p16
GRU_LIBRARY ?=
GRU_INCLUDE ?= ../tflite_qnn_prototype/include
MODEL_LIBRARY ?= $(GRU_LIBRARY)
MODEL_INCLUDE ?= $(GRU_INCLUDE)
MODEL_PROFILE ?= model_profiles/tinygru_p16/model_profile.json
MODEL_PROFILE_DIR := $(dir $(MODEL_PROFILE))
PREPROCESSING_SOURCE := $(MODEL_PROFILE_DIR)traffic_p16_deployment_config.c
PREPROCESSING_HEADER := $(MODEL_PROFILE_DIR)traffic_p16_deployment_config.h
PROFILE_GEN_DIR := $(OUT)/generated_profile
PROFILE_STAMP := $(PROFILE_GEN_DIR)/.stamp

ifneq ($(MODEL_LIBRARY),)
MODEL_FLAGS = -DTRAFFIC_AI_WITH_MODEL -I$(MODEL_INCLUDE)
MODEL_LINK = -L$(dir $(MODEL_LIBRARY)) -l:$(notdir $(MODEL_LIBRARY)) -Wl,-rpath,'$$ORIGIN'
APP_MODEL_LINK = -Wl,-rpath-link,$(dir $(MODEL_LIBRARY))
endif

.PHONY: all combined test integration-test profile
all: combined

# A single public SDK for the current production pair: Random Forest + the
# two-input TinyGRU P16.  The old one-input GRU is intentionally not linked.
# It keeps the deployment to libtraffic_ai.so + its one P16 model dependency.
combined: $(OUT)/libtraffic_ai.so $(OUT)/app_arm64

$(OUT):
	mkdir -p $(OUT)

$(PROFILE_STAMP): $(MODEL_PROFILE) tools/generate_model_profile.py | $(OUT)
	$(PYTHON) tools/generate_model_profile.py --profile $(MODEL_PROFILE) --output-dir $(PROFILE_GEN_DIR)
	touch $@

profile: $(PROFILE_STAMP)

$(OUT)/libtraffic_ai.so: traffic_p16.c traffic_p16.h model_contract.h model_runtime.c model_runtime.h \
	$(PREPROCESSING_SOURCE) $(PREPROCESSING_HEADER) \
	../traffic_models/traffic_models.c ../traffic_models/traffic_models.h \
	../traffic_models/rf_feature_extractor.c ../traffic_models/vendor/rf_model.c \
	traffic_ai_profile.c traffic_ai.h exports_combined.map Makefile \
	$(MODEL_LIBRARY) $(PROFILE_STAMP) | $(OUT)
	$(CC) $(CFLAGS) -I$(PROFILE_GEN_DIR) $(MODEL_FLAGS) -fPIC -shared \
		../traffic_models/traffic_models.c ../traffic_models/rf_feature_extractor.c \
		../traffic_models/vendor/rf_model.c model_runtime.c traffic_ai_profile.c traffic_p16.c \
		$(PREPROCESSING_SOURCE) \
		$(PROFILE_GEN_DIR)/traffic_model_profile.c $(MODEL_LINK) -lm \
		-Wl,--gc-sections -Wl,--version-script=exports_combined.map \
		-Wl,-soname,libtraffic_ai.so -o $@
	$(STRIP) --strip-unneeded $@

$(OUT)/app_arm64: examples/p16_smoke.c traffic_p16.h $(PREPROCESSING_HEADER) Makefile $(OUT)/libtraffic_ai.so
	$(CC) $(CFLAGS) examples/p16_smoke.c -L$(OUT) -ltraffic_ai $(APP_MODEL_LINK) \
		-Wl,-rpath,'$$ORIGIN:$$ORIGIN/../lib' -o $@
	$(STRIP) --strip-unneeded $@

test: $(PROFILE_STAMP)
	$(CC) $(CFLAGS) -I$(PROFILE_GEN_DIR) model_runtime.c traffic_p16.c \
		$(PREPROCESSING_SOURCE) \
		$(PROFILE_GEN_DIR)/traffic_model_profile.c tests/test_preprocessing.c \
		-lm -o /tmp/traffic_p16_test
	/tmp/traffic_p16_test

integration-test:
	rm -rf /tmp/traffic_ai_integration
	mkdir -p /tmp/traffic_ai_integration/model
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-I../tflite_qnn_prototype/include tests/fake_ai_model.c \
		-o /tmp/traffic_ai_integration/model/libfake_model.so
	$(MAKE) combined OUT=/tmp/traffic_ai_integration/sdk \
		MODEL_PROFILE=model_profiles/tinygru_p16/model_profile.json \
		MODEL_LIBRARY=/tmp/traffic_ai_integration/model/libfake_model.so \
		MODEL_INCLUDE=../tflite_qnn_prototype/include STRIP=true
	cp /tmp/traffic_ai_integration/model/libfake_model.so /tmp/traffic_ai_integration/sdk/
	LD_LIBRARY_PATH=/tmp/traffic_ai_integration/sdk \
		/tmp/traffic_ai_integration/sdk/app_arm64 | \
		grep -E 'model=tinygru-p16-v1|class=RTVideo'
