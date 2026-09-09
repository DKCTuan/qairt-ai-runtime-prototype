#include "traffic_models.h"
#include <assert.h>
#include <math.h>
extern int predict_forest(float *, float *);
int main(void) {
    float rf[11]={0,26.9784f,832.711f,100,60,19.7368f,0,0,0.000524f,0.995556f,0};
    float ref[5]; traffic_result r;
    int label=predict_forest(rf,ref);
    assert(traffic_rf_predict(rf,11,&r)==0 && r.label==label);
    for(int i=0;i<5;i++) assert(r.scores[i]==ref[i]);
    assert(traffic_rf_predict(rf,10,&r)==TRAFFIC_INVALID_INPUT);
    rf[0]=NAN; assert(traffic_rf_predict(rf,11,&r)==TRAFFIC_INVALID_INPUT);
    uint64_t timestamps[TRAFFIC_GRU_PACKETS];
    uint32_t lengths[TRAFFIC_GRU_PACKETS];
    float extracted[TRAFFIC_RF_FEATURES];
    float seq[270];
    for (int i=0;i<TRAFFIC_GRU_PACKETS;i++) {
        timestamps[i]=(uint64_t)i*20000; lengths[i]=60;
    }
    assert(traffic_rf_extract(timestamps,lengths,TRAFFIC_GRU_PACKETS,extracted)==TRAFFIC_OK);
    assert(fabsf(extracted[0]) < 1e-6f);       /* frac_mid */
    assert(fabsf(extracted[1]-1.0f) < 1e-5f);  /* p95 / p05 */
    assert(fabsf(extracted[2]-60.0f) < 1e-5f); /* mean length */
    assert(fabsf(extracted[8]-0.02f) < 1e-6f); /* max IAT seconds */
    assert(traffic_rf_extract(timestamps,lengths,89,extracted)==TRAFFIC_INVALID_INPUT);
    timestamps[20]=timestamps[19]-1;
    assert(traffic_rf_extract(timestamps,lengths,TRAFFIC_GRU_PACKETS,extracted)==TRAFFIC_INVALID_INPUT);
    timestamps[20]=400000;
    int8_t directions[TRAFFIC_GRU_PACKETS];
    for(int i=0;i<TRAFFIC_GRU_PACKETS;i++) directions[i]=(i&1)?-1:1;
    assert(traffic_gru_prepare_packets(timestamps,directions,lengths,
                                       TRAFFIC_GRU_PACKETS,seq)==TRAFFIC_OK);
    assert(fabsf(seq[0] - ((0.0f - 0.005491858348250389f) / 0.07258545607328415f)) < 1e-5f);
    assert(seq[1] == 1.0f);
    directions[0]=0;
    assert(traffic_gru_prepare_packets(timestamps,directions,lengths,
                                       TRAFFIC_GRU_PACKETS,seq)==TRAFFIC_INVALID_INPUT);
    for(int i=0;i<270;i+=3) {
        seq[i]=0.005491858348250389f; seq[i+1]=-1; seq[i+2]=626.3450317382812f;
    }
    assert(traffic_gru_prepare(seq,270,seq)==0);
    for(int i=0;i<270;i+=3) assert(seq[i]==0 && seq[i+1]==-1 && seq[i+2]==0);
    assert(traffic_gru_init()==TRAFFIC_MODEL_UNAVAILABLE);
    assert(traffic_gru_predict(seq,270,&r)==TRAFFIC_MODEL_UNAVAILABLE && r.label==-1);
    traffic_gru_deinit();
    return 0;
}
