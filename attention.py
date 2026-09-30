import numpy as np 
 
def softmax(x): 
    exp_x = np.exp( 
        x - np.max(x, axis=-1, keepdims=True) 
    ) 
    return exp_x / np.sum( 
        exp_x, 
        axis=-1, 
        keepdims=True 
    ) 
 
def calculate_attention(X): 
    d_model = X.shape[1] 
    d_k = 64 
 
    np.random.seed(42) 
 
    WQ = np.random.randn(d_model, d_k) 
    WK = np.random.randn(d_model, d_k) 
    WV = np.random.randn(d_model, d_k) 
 
    Q = X @ WQ 
    K = X @ WK 
    V = X @ WV 
 
    scores = Q @ K.T 
 
    scaled_scores = scores / np.sqrt(d_k) 
 
    attention_weights = softmax(scaled_scores) 
 
    word_scores = attention_weights.mean(axis=0) 
 
    return word_scores