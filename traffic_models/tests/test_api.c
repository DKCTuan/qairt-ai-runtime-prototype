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
    float seq[270];
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
