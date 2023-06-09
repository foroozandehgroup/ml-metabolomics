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

        x = x.copy()
        y = y.copy()

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

        # orthogonal
        self.T_ortho = T_ortho
        self.P_ortho = P_ortho
        self.W_ortho = W_ortho

        # predictive
        self.T = T
        self.P = P
        self.coef = tw * C[:, np.newaxis]

        self.C = C

        tw = tw.reshape(-1, 1)
        self.W = tw
    
    def fit_transform(self, x: np.ndarray, y: np.ndarray):
        
        self.fit(x, y)

        return self.T, self.C

    def inverse_transform(self, x_scores: np.ndarray):
        
        return x_scores @ self.P.T

    def transform(self, x_test: np.ndarray):
        
        return x_test @ self.P

    def predict(self, x: np.ndarray, return_scores: bool=False):
        
        x = x.copy()
        x = self.correct(x)

        coef = self.coef[self.n_components - 1]
        y = np.dot(x, coef)

        if return_scores:
            return y, np.dot(x, self.W)
        
        return y

    def correct(self, x: np.ndarray, return_scores: bool=False):

        x_corr = x.copy()

        if x_corr.ndim == 1:
            t = np.empty(self.n_components)
            for nc in range(self.n_components):
                t_ = np.dot(x_corr, self.W_ortho[:, nc])
                x_corr -= t_ * self.P_ortho[:, nc]
                t[nc] = t_
        
        else:
            n, c = x_corr.shape
            t = np.empty((n, self.n_components))
            for nc in range(self.n_components):
                t_ = np.dot(x_corr, self.W_ortho[:, nc])
                x_corr -= t_[:, np.newaxis] * self.P_ortho[:, nc]
                t[:, nc] = t_
        
        if return_scores:
            return x_corr, t
        
        return x_corr
