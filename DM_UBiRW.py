import math
import numpy as np
from dm_cal import getSimilarMatrix
from dm_cal import row_norm
from DM_train import DM_method


class UBiRW(DM_method):
    def __init__(self, A, SD, SV, **argv):
        self.SD = SD
        self.SV = SV
        self.argv = {'alpha':0.45, 'l1':1, 'l2':2, 'dlalpha':0.1}
        self.argv.update(argv)
        super(UBiRW, self).__init__(A)

    def fun(self, A_):
        B = A_
        alpha = self.argv['alpha']
        l1 = self.argv['l1']
        l2 = self.argv['l2']
        dlalpha = self.argv['dlalpha']
        vlalpha = self.argv['vlalpha']
        lam = self.argv['lam']
        W_sm_vi = B.copy()

        sm_sim = self.SD
        vi_sim = self.SV
        # sm_sim = getSimilarMatrix(A_, 1)
        # vi_sim = getSimilarMatrix(A_.T, 1)
        sm_sim = dlalpha*self.SD + (1-dlalpha)*getSimilarMatrix(A_, lam)
        vi_sim = vlalpha*self.SV + (1-vlalpha)*getSimilarMatrix(A_.T, lam)

        cn = 0

        P_sm_vi_inl = (1/B.sum()) * B
        
        DV_2 = np.diag(np.sqrt(1/np.sum(vi_sim, axis=1)))
        DD_2 = np.diag(np.sqrt(1/np.sum(sm_sim, axis=1)))

        DV = np.diag(np.sum(vi_sim, axis=1))
        DD = np.diag(np.sum(sm_sim, axis=1))

        LV = DV_2 @ (DV - vi_sim) @ DV_2
        LD = DD_2 @ (DD - sm_sim) @ DD_2
        FV = LV
        FD = LD

        FV = vi_sim @ np.linalg.pinv(vi_sim + 0.01 * LV @ vi_sim)
        FD = sm_sim @ np.linalg.pinv(sm_sim + 0.01 * LD @ sm_sim)

        M = max(l1, l2)
        n1 = 0
        n2 = 0
        W1 = np.zeros(B.shape)
        W2 = np.zeros(B.shape)

        while(cn <= M):
            W_sm_vi = 1/W_sm_vi.sum() * W_sm_vi
            if(n1 <= l1):
                W1 = alpha*(sm_sim @ P_sm_vi_inl) + (1-alpha)*P_sm_vi_inl
                W1 = W1 @ FV
                # W1 = FD @ W1
                n1 += 1
            if(n2 <= l2):
                W2 = alpha*(P_sm_vi_inl @ vi_sim) + (1-alpha)*P_sm_vi_inl
                W2 = FD @ W2
                # W2 = W2 @ FV
                n2 += 1
            W_sm_vi = (W1 + W2)/2

            cn += 1
        return W_sm_vi

