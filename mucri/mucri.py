'''
Created on 2017-2-19

@author: Jun Chen
'''

import numpy as np
from evaluate import get_acc
from evaluate import get_auc
from dataset import INRIAFeed
from dataset import AestheticFeed


def safe_logistic(x):
    """
    a safe logistic function
    """
    if x >= 0: return 1. / (1 + np.exp(-x))
    else: return 1 - 1. / (np.exp(x) + 1)


class MuCriPreference:
    """
    the family of MuCri models
    """
    
    def __init__(self, model='hybrid'):
        if model == 'hybrid':
            self.fit = self._fit_hybrid
            self.predict_base = self._predict_hybrid
        elif model == 'hybrid mean':
            self.fit = self._fit_hybrid_mean
            self.predict_base = self._predict_hybrid_mean
        elif model == 'hybrid max':
            self.fit = self._fit_hybrid_max
            self.predict_base = self._predict_hybrid_max
        elif model == 'latent':
            self.fit = self._fit_latent
            self.predict_base = self._predict_latent
        elif model == 'content':
            self.fit = self._fit_content
            self.predict_base = self._predict_content
        else:
            raise ValueError('undefined model.')
    

    def _fit_hybrid(self, X, contents, U, I, Fl=2, Fc=2, Dl=10, Dc=6, k=100, \
        learning_rate=.02, gamma=.01, lamda=.01, max_iters=300, tol=1e-3):
        """
            fit the H-MuCri-Prod model
            X: comparisons
            C: content features
            U: user number
            I: item number
            Fl: F of L-MuCri
            Fc: F of C-MuCri
            Dl: latent attributes number
            Dc: content feature number
            k: latent feature dimension of L-MuCri
        """
        self.Fl, self.Fc, self.Dl, self.Dc = Fl, Fc, Dl, Dc
        print 'Training H-MuCri-Prod ...'
        ultensor, vltensor = self._init_latent(U, I, Dl, k)
        cfdims, uctensor = self._init_content(U, contents)
        ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
        uctensor_best = [data_mat.copy() for data_mat in uctensor]
        losslist = [np.finfo(float).max]
        v_tmp = X[0][0]
        for it in xrange(max_iters):
            #update vectors
            np.random.shuffle(X)
            for a, b, u in X:
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                
                #update features
                #L-MuCri
                self._update_latent_inplace(ultensor, vltensor, u, a, b, prob_luab, cand_lt, learning_rate, lamda)
                #C-MuCri
                self._update_content_inplace(uctensor, contents, u, a, b, Dc, cand_ct, prob_cuab, learning_rate, gamma)
            #compute loss
            curr_ll = 0
            for a, b, u in X: 
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                #H-MuCri
                prob_uab = prob_luab * prob_cuab
                curr_ll += -np.log(prob_uab) 
                    
            print 'iter: %d, loss: %0.3f' % (it, curr_ll), vltensor[v_tmp, 0, :5]
            if curr_ll > max(losslist[-3:]):
                break
            else: 
                if curr_ll < min(losslist): 
                    ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
                    uctensor_best = [data_mat.copy() for data_mat in uctensor]
                losslist.append(curr_ll)
            
        print 'Training finished.'
        self.ultensor, self.vltensor, self.uctensor, self.cfdims = ultensor_best, vltensor_best, uctensor_best, cfdims
    
    
    def _fit_hybrid_mean(self, X, contents, U, I, Fl=2, Fc=2, Dl=10, Dc=6, k=100, \
        learning_rate=.02, gamma=.01, lamda=.01, max_iters=300, tol=1e-3):
        """
        fit the H-MuCri-Mean model
        """
        self.Fl, self.Fc, self.Dl, self.Dc = Fl, Fc, Dl, Dc
        print 'Training H-MuCri-Mean ...'
        ultensor, vltensor = self._init_latent(U, I, Dl, k)
        cfdims, uctensor = self._init_content(U, contents)
        ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
        uctensor_best = [data_mat.copy() for data_mat in uctensor]
        losslist = [np.finfo(float).max]
        v_tmp = X[0][0]
        for it in xrange(max_iters):
            #update vectors
            np.random.shuffle(X)
            for a, b, u in X:
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                score_lt = self._get_preference_score(pref_lua, pref_lub, cand_lt)
                prob_luab = safe_logistic(score_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                score_ct = self._get_preference_score(pref_cua, pref_cub, cand_ct)
                prob_cuab = safe_logistic(score_ct)
                
                #update features
                self._update_hybrid_mean_inplace(ultensor, vltensor, uctensor, contents, u, a, b, (prob_luab + prob_cuab) / 2, cand_lt, cand_ct, score_lt, score_ct, \
                                                 prob_luab, prob_cuab, learning_rate, lamda, gamma)
            #compute loss
            curr_ll = 0
            for a, b, u in X: 
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                #Mean H-MuCri
                prob_uab = (prob_luab + prob_cuab) / 2
                curr_ll += -np.log(prob_uab) 
                    
            print 'iter: %d, loss: %0.3f' % (it, curr_ll), vltensor[v_tmp, 0, :5]
            if curr_ll > max(losslist[-3:]):
                break
            else: 
                if curr_ll < min(losslist): 
                    ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
                    uctensor_best = [data_mat.copy() for data_mat in uctensor]
                losslist.append(curr_ll)
            
        print 'Training finished.'
        self.ultensor, self.vltensor, self.uctensor, self.cfdims = ultensor_best, vltensor_best, uctensor_best, cfdims
        

    def _fit_hybrid_max(self, X, contents, U, I, Fl=2, Fc=2, Dl=10, Dc=6, k=100, \
        learning_rate=.02, gamma=.01, lamda=.01, max_iters=300, tol=1e-3):
        """
        fit the H-MuCri-Max model
        """
        self.Fl, self.Fc, self.Dl, self.Dc = Fl, Fc, Dl, Dc
        print 'Training H-MuCri-Max ...'
        ultensor, vltensor = self._init_latent(U, I, Dl, k)
        cfdims, uctensor = self._init_content(U, contents)
        ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
        uctensor_best = [data_mat.copy() for data_mat in uctensor]
        losslist = [np.finfo(float).max]
        v_tmp = X[0][0]
        for it in xrange(max_iters):
            #update vectors
            np.random.shuffle(X)
            for a, b, u in X:
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                
                #update features
                if prob_luab >= prob_cuab:
                    self._update_latent_inplace(ultensor, vltensor, u, a, b, prob_luab, cand_lt, learning_rate, lamda)
                else:
                    self._update_content_inplace(uctensor, contents, u, a, b, Dc, cand_ct, prob_cuab, learning_rate, gamma)
            #compute loss
            curr_ll = 0
            for a, b, u in X: 
                #L-MuCri
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #C-MuCri
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                #H-MuCri
                prob_uab = max(prob_luab, prob_cuab)
                curr_ll += -np.log(prob_uab) 
                    
            print 'iter: %d, loss: %0.3f' % (it, curr_ll), vltensor[v_tmp, 0, :5]
            if curr_ll > max(losslist[-3:]):
                break
            else: 
                if curr_ll < min(losslist): 
                    ultensor_best, vltensor_best = ultensor.copy(), vltensor.copy()
                    uctensor_best = [data_mat.copy() for data_mat in uctensor]
                losslist.append(curr_ll)
            
        print 'Training finished.'
        self.ultensor, self.vltensor, self.uctensor, self.cfdims = ultensor_best, vltensor_best, uctensor_best, cfdims
    

    def _fit_latent(self, X, U, I, Fl=2, Dl=10, k=100, learning_rate=.02, \
        lamda=.01, max_iters=300, tol=1e-3):
        """
        fit the L-MuCri model
        """
        self.Fl, self.Dl = Fl, Dl
        print 'In training ...'
        prev_ll = np.finfo(float).max #loss
        ultensor, vltensor = self._init_latent(U, I, Dl, k)
        v_tmp = X[0][0]
        for it in xrange(max_iters):
            #update vectors
            np.random.shuffle(X)
            for a, b, u in X:
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                #update features               
                self._update_latent_inplace(ultensor, vltensor, u, a, b, prob_luab, cand_lt, learning_rate, lamda)
            #compute loss
            curr_ll = 0
            for a, b, u in X: 
                pref_lua = self._get_preference_latent(ultensor, vltensor, u, a)
                pref_lub = self._get_preference_latent(ultensor, vltensor, u, b)
                cand_lt = self._get_topmost_indices(pref_lua, pref_lub, Fl)
                prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
                curr_ll += -np.log(prob_luab) 
                    
            print 'iter: %d, loss: %0.3f' % (it, curr_ll)
            if prev_ll - curr_ll < tol: break
            else: prev_ll = curr_ll
            
        print 'Training finished.'
        self.ultensor, self.vltensor = ultensor, vltensor
    
    
    def _fit_content(self, X, contents, U, I, Fc=4, Dc=6, k=100, learning_rate=.02, \
        gamma=.01, max_iters=300, tol=1e-3):
        """
        fit the C-MuCri model
        """
        self.Fc, self.Dc = Fc, Dc
        print 'In training ...'
        #prev_ll = np.finfo(float).max #loss
        cfdims, uctensor = self._init_content(U, contents)
        uctensor_best = [data_mat.copy() for data_mat in uctensor]
        losslist = [np.finfo(float).max]
        for it in xrange(max_iters):
            #update vectors
            np.random.shuffle(X)
            for a, b, u in X:
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                #update features
                self._update_content_inplace(uctensor, contents, u, a, b, Dc, cand_ct, prob_cuab, learning_rate, gamma)
            #compute loss
            curr_ll = 0
            for a, b, u in X: 
                pref_cua = self._get_preference_content(uctensor, contents, u, a)
                pref_cub = self._get_preference_content(uctensor, contents, u, b)
                cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, Fc)
                prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
                curr_ll += -np.log(prob_cuab)     
            print 'iter: %d, loss: %0.3f' % (it, curr_ll), uctensor[0][0][:5]
            if curr_ll > max(losslist[-3:]):
                break
            else: 
                if curr_ll < min(losslist): uctensor_best = [data_mat.copy() for data_mat in uctensor]
                losslist.append(curr_ll)
            
        print 'Training finished.'
        self.uctensor, self.cfdims = uctensor_best, cfdims


    def _get_topmost_indices(self, pa, pb, F):
        return set(np.argsort(-pa)[:F]) | set(np.argsort(-pb)[:F])


    def _get_preference_content(self, uctensor, content, u, v):
        return np.array([mat[u].dot(content[x][v]) for x, mat in enumerate(uctensor)])


    def _get_preference_latent(self, ultensor, vltensor, u, v):
        return np.sum(ultensor[u] * vltensor[v], axis=1)


    def _get_preference_prob(self, pa, pb, cand):
        return safe_logistic(np.mean([pa[x] - pb[x] for x in cand]))
    
    
    def _get_preference_score(self, pa, pb, cand):
        return np.mean([pa[x] - pb[x] for x in cand])
    

    def _update_content_inplace(self, uctensor, contents, u, a, b, Dc, cand, prob_uab, learning_rate, gamma):
        for x in xrange(Dc):
            if x in cand:
                uctensor[x][u] = (1 - learning_rate * gamma) * uctensor[x][u] \
                                + learning_rate * (1 - prob_uab) * (contents[x][a] - contents[x][b]) / len(cand)
            else:
                uctensor[x][u] = (1 - learning_rate * gamma) * uctensor[x][u]


    def _update_latent_inplace(self, ultensor, vltensor, u, a, b, prob_uab, cand, learning_rate, lamda):
        u_new = (1 - learning_rate * lamda) * ultensor[u]
        va_new = (1 - learning_rate * lamda) * vltensor[a]
        vb_new = (1 - learning_rate * lamda) * vltensor[b]
        for x in cand:
            u_new[x]  += learning_rate * (1 - prob_uab) * (vltensor[a][x] - vltensor[b][x]) / len(cand)
            va_new[x] += learning_rate * (1 - prob_uab) * (ultensor[u][x] / len(cand))
            vb_new[x] += learning_rate * (1 - prob_uab) * (- ultensor[u][x] / len(cand))
        ultensor[u], vltensor[a], vltensor[b] = u_new, va_new, vb_new


    def _update_hybrid_mean_inplace(self, ultensor, vltensor, uctensor, contents, u, a, b, prob_uab, \
        cand_lt, cand_ct, r_lt_uab, r_ct_uab, prob_lt_uab, prob_ct_uab, learning_rate, lamda, gamma):
        #update latent
        const_lt = prob_lt_uab**2 * np.exp(-r_lt_uab) / (2 * prob_uab)
        u_new = (1 - learning_rate * lamda) * ultensor[u]
        va_new = (1 - learning_rate * lamda) * vltensor[a]
        vb_new = (1 - learning_rate * lamda) * vltensor[b]
        for x in cand_lt:
            u_new[x]  += learning_rate * const_lt * (vltensor[a][x] - vltensor[b][x]) / len(cand_lt)
            va_new[x] += learning_rate * const_lt * (ultensor[u][x] / len(cand_lt))
            vb_new[x] += learning_rate * const_lt * (- ultensor[u][x] / len(cand_lt))
        ultensor[u], vltensor[a], vltensor[b] = u_new, va_new, vb_new
        
        #update content
        const_ct = prob_ct_uab**2 * np.exp(-r_ct_uab) / (2 * prob_uab)
        for x in xrange(len(uctensor)):
            if x in cand_ct:
                uctensor[x][u] = (1 - learning_rate * gamma) * uctensor[x][u] \
                                + learning_rate * const_ct * (contents[x][a] - contents[x][b]) / len(cand_ct)
            else:
                uctensor[x][u] = (1 - learning_rate * gamma) * uctensor[x][u]


    def _init_latent(self, U, I, Dl, k):
        ultensor = np.random.normal(0., .1, (U, Dl, k))
        vltensor = np.random.normal(0., .1, (I, Dl, k))
        return ultensor, vltensor


    def _init_content(self, U, contents):
        cfdims = [len(f[1]) for f in contents] #content feature dimensions
        uctensor = [np.random.normal(0., .001, (U, dim)) for dim in cfdims]
        return cfdims, uctensor


    def _predict_hybrid(self, a, b, u, contents):
        #L-MuCri
        pref_lua = self._get_preference_latent(self.ultensor, self.vltensor, u, a)
        pref_lub = self._get_preference_latent(self.ultensor, self.vltensor, u, b)
        cand_lt = self._get_topmost_indices(pref_lua, pref_lub, self.Fl)
        prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
        prob_luba = self._get_preference_prob(pref_lub, pref_lua, cand_lt)
        
        #C-MuCri
        pref_cua = self._get_preference_content(self.uctensor, contents, u, a)
        pref_cub = self._get_preference_content(self.uctensor, contents, u, b)
        cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, self.Fc)
        prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
        prob_cuba = self._get_preference_prob(pref_cub, pref_cua, cand_ct)
                
        if prob_luab * prob_cuab >= prob_luba * prob_cuba: return [a, b]
        else: return [b, a]
    
    
    def _predict_hybrid_mean(self, a, b, u, contents):
        #L-MuCri
        pref_lua = self._get_preference_latent(self.ultensor, self.vltensor, u, a)
        pref_lub = self._get_preference_latent(self.ultensor, self.vltensor, u, b)
        cand_lt = self._get_topmost_indices(pref_lua, pref_lub, self.Fl)
        prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
        prob_luba = self._get_preference_prob(pref_lub, pref_lua, cand_lt)
        
        #C-MuCri
        pref_cua = self._get_preference_content(self.uctensor, contents, u, a)
        pref_cub = self._get_preference_content(self.uctensor, contents, u, b)
        cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, self.Fc)
        prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
        prob_cuba = self._get_preference_prob(pref_cub, pref_cua, cand_ct)
                
        if prob_luab + prob_cuab >= prob_luba + prob_cuba: return [a, b]
        else: return [b, a]
    
    
    def _predict_hybrid_max(self, a, b, u, contents):
        #L-MuCri
        pref_lua = self._get_preference_latent(self.ultensor, self.vltensor, u, a)
        pref_lub = self._get_preference_latent(self.ultensor, self.vltensor, u, b)
        cand_lt = self._get_topmost_indices(pref_lua, pref_lub, self.Fl)
        prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
        prob_luba = self._get_preference_prob(pref_lub, pref_lua, cand_lt)
        
        #C-MuCri
        pref_cua = self._get_preference_content(self.uctensor, contents, u, a)
        pref_cub = self._get_preference_content(self.uctensor, contents, u, b)
        cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, self.Fc)
        prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
        prob_cuba = self._get_preference_prob(pref_cub, pref_cua, cand_ct)
                
        if max(prob_luab, prob_cuab) >= max(prob_luba, prob_cuba): return [a, b]
        else: return [b, a]
        
    
    def _predict_latent(self, a, b, u, contents):
        pref_lua = self._get_preference_latent(self.ultensor, self.vltensor, u, a)
        pref_lub = self._get_preference_latent(self.ultensor, self.vltensor, u, b)
        cand_lt = self._get_topmost_indices(pref_lua, pref_lub, self.Fl)
        prob_luab = self._get_preference_prob(pref_lua, pref_lub, cand_lt)
        if prob_luab >= .5: return [a, b]
        else: return [b, a]
    
    
    def _predict_content(self, a, b, u, contents):
        pref_cua = self._get_preference_content(self.uctensor, contents, u, a)
        pref_cub = self._get_preference_content(self.uctensor, contents, u, b)
        cand_ct  = self._get_topmost_indices(pref_cua, pref_cub, self.Fc)
        prob_cuab = self._get_preference_prob(pref_cua, pref_cub, cand_ct)
        if prob_cuab >= .5: return [a, b]
        else: return [b, a]


    def predict(self, X, contents=None):
        return [self.predict_base(a, b, u, contents) for a, b, u in X]


    def analyze_topmost_selections_latent(self):
        user_selections_single = np.zeros((self.ultensor.shape[0], self.ultensor.shape[1]), dtype=float)
        item_selections_single = np.zeros((self.vltensor.shape[0], self.vltensor.shape[1]), dtype=float)
        selections_all         = np.zeros(self.ultensor.shape[1], dtype=float)
        for u in xrange(user_selections_single.shape[0]):
            for v in xrange(1, item_selections_single.shape[0]):
                pref = self._get_preference_latent(self.ultensor, self.vltensor, u, v)
                idx = np.argmax(pref)
                user_selections_single[u, idx] += 1
                item_selections_single[v, idx] += 1
                selections_all[idx]            += 1
        print 'User selections ...'
        user_selections_single /= np.sum(user_selections_single, axis=1)[:, np.newaxis]
        for u in xrange(user_selections_single.shape[0]):
            print 'User %d: ' % u, user_selections_single[u]
        print 'Item selections ...'
        item_selections_single /= np.sum(item_selections_single, axis=1)[:, np.newaxis]
        for v in xrange(item_selections_single.shape[0]):
            print 'Item %d: ' % v, item_selections_single[v]
        print 'Overall selections ...'
        selections_all /= np.sum(selections_all)
        for c in xrange(len(selections_all)):
            print 'Item %d: ' % c, selections_all[c]
    
    
    def analyze_topmost_selections_content(self, contents):
        user_selections_single = np.zeros((self.uctensor[0].shape[0], len(self.uctensor)), dtype=float)
        item_selections_single = np.zeros((contents[0].shape[0], len(contents)), dtype=float)
        selections_all         = np.zeros(len(contents), dtype=float)
        for u in xrange(user_selections_single.shape[0]):
            for v in xrange(1, item_selections_single.shape[0]):
                pref = self._get_preference_content(self.uctensor, contents, u, v)
                idx = np.argmax(pref)
                user_selections_single[u, idx] += 1
                item_selections_single[v, idx] += 1
                selections_all[idx]            += 1
        print 'User selections ...'
        user_selections_single /= np.sum(user_selections_single, axis=1)[:, np.newaxis]
        for u in xrange(user_selections_single.shape[0]):
            print 'User %d: ' % u, user_selections_single[u]
        print 'Item selections ...'
        item_selections_single /= np.sum(item_selections_single, axis=1)[:, np.newaxis]
        for v in xrange(item_selections_single.shape[0]):
            print 'Item %d: ' % v, item_selections_single[v]
        print 'Overall selections ...'
        selections_all /= np.sum(selections_all)
        for c in xrange(len(selections_all)):
            print 'Item %d: ' % c, selections_all[c]


if __name__ == '__main__':
    
    
    feed = INRIAFeed()
    #feed = AestheticFeed()
    
    Xtrain, Xtest = feed.generate_data()
    
    model = MuCriPreference(model='latent')
    
    run = 1
    acc_array, auc_array = [], []
    for train, test in zip(Xtrain, Xtest):
        model.fit(X=train, U=feed.U, I=feed.I, Fl=5, Dl=6, k=100, learning_rate=0.02, \
            lamda=.01, max_iters=300, tol=-1)
        X_pred = model.predict(test, contents)
        acc = get_acc(x_pred=X_pred, x_true=test)
        auc = get_auc(x_pred=X_pred, x_true=test)
        print 'Run #\%d, accuracy:%.5f, auc:%.5f' % (run, acc, auc)
        run += 1
        acc_array.append(acc)
        auc_array.append(auc)
    
    print 'Accuracy:'
    for v in acc_array: print '\t%.5f' % v
    print 'Average accuracy:%.5f' % np.mean(acc_array)
    
    print 'AUC:'
    for v in auc_array: print '\t%.5f' % v
    print 'Average AUC:%.5f' % np.mean(auc_array)
    
    print 'Done!'
    



