from dataclasses import dataclass
import numpy as np
import numpy.linalg as la

from Metabolomics_ML.algorithms.nipals import nipals

@dataclass
class OPLS:
    n_components: int

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits OPLS model using NIPALS algorithm
        """

        n, p = x.shape
        npc = min(n, p)

        if self.n_components < npc:
            npc = self.n_components
        
        # initialise matrices
        T_ortho, P_ortho, W_ortho = np.empty((n, npc)), np.empty((p, npc)), np.empty((p, npc))
        T, P, C = np.empty((n, npc)), np.empty((p, npc)), np.empty(npc)

        tw = np.dot(y, x) / np.dot(y, y)
        tw /= la.norm(tw)

        tp = np.dot(x, tw)

        # get components
        w, u, c, t = nipals(x, y)
        p = np.dot(t, x) / np.dot(t, t)

        for nc in range(npc):
            w_ortho = p - (np.dot(tw, p) * tw)
            w_ortho /= la.norm(w_ortho)

            t_ortho = np.dot(x, w_ortho)

            p_ortho = np.dot(t_ortho, x) / np.dot(t_ortho, t_ortho)

            x -= t_ortho[:, np.newaxis] * p_ortho
            
            T_ortho[:, nc] = t_ortho
            P_ortho[:, nc] = p_ortho
            W_ortho[:, nc] = w_ortho

            tp -= t_ortho * np.dot(p_ortho, tw)

            T[:, nc] = tp
            C[nc] = np.dot(y, tp) / np.dot(tp, tp)

            # next component
            w, u, c, t = nipals(x, y)

            p = np.dot(t, x) / np.dot(t, t)
            P[:, nc] = p

        # orthogonal scores
        self.T_ortho = T_ortho
        self.P_ortho = P_ortho
        self.W_ortho = W_ortho

        # covariate weights
        self._W_cov = tw

        # predictive scores
        self.T = T
        self.P = P
        self.C = C
    
    def predict(self):
        pass
