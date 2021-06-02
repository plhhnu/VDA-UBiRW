import numpy as np
from dm_cal import getSimilarMatrix
from dm_cal import row_norm
from DM_train import DM_method


class BiRW(DM_method):
    def __init__(self, A, SD, SV, **argv):
        self.SD = SD
        self.SV = SV
        self.argv = {'alpha':0.9, 'M':10}
        self.argv.update(argv)
        super(BiRW, self).__init__(A)

    def fun(self, A_):
        # Nd, Nv = A_.shape
        B = A_
        sm_sim = self.SD
        vi_sim = self.SV
        W_sm_vi = B.copy()
        
        alpha = self.argv['alpha']
        M = self.argv['M']
        dlalpha = self.argv['dlalpha']
        vlalpha = self.argv['vlalpha']
        lam = self.argv['lam']


        sm_sim = self.SD
        vi_sim = self.SV
        # sm_sim = getSimilarMatrix(A_, 1)
        # vi_sim = getSimilarMatrix(A_.T, 1)
        sm_sim = dlalpha*self.SD + (1-dlalpha)*getSimilarMatrix(A_, lam)
        vi_sim = vlalpha*self.SV + (1-vlalpha)*getSimilarMatrix(A_.T, lam)

        t = 0
        cn = 0

        P_sm_vi_inl = (1/B.sum()) * B
        W1 = np.zeros(B.shape)
        W2 = np.zeros(B.shape)

        while(cn <= M):
            if(t == 0):
                W1 = alpha*(sm_sim @ P_sm_vi_inl) + (1-alpha)*P_sm_vi_inl
                W2 = alpha*(P_sm_vi_inl @ vi_sim) + (1-alpha)*P_sm_vi_inl
                W_sm_vi = (W1 + W2)/2
                t += 1
            else:
                W_sm_vi = 1/W_sm_vi.sum() * W_sm_vi
                W1 = alpha*(sm_sim @ P_sm_vi_inl) + (1-alpha)*P_sm_vi_inl
                W2 = alpha*(P_sm_vi_inl @ vi_sim) + (1-alpha)*P_sm_vi_inl
                W_sm_vi = (W1 + W2)/2
            
            cn += 1
        return W_sm_vi


