#include "traffic_ai.h"

#include "traffic_model_profile.h"

const traffic_model_contract_t *traffic_ai_model_contract(void)
{
    return &traffic_model_profile_contract;
}

const char *traffic_ai_model_id(void)
{
    return traffic_model_profile_contract.model_id;
}

const char *traffic_ai_preprocessing_id(void)
{
    return traffic_model_profile_contract.preprocessing_id;
}
