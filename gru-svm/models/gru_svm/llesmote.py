import tensorflow as tf

def LLE(input_, row, num_classes):
    var_total = tf.constant(0.0)
    W = tf.zeros((tf.shape(input_[0])))
    def find_nearest(input_):
        # Nearest Neighbor calculation using L1 Distance
        # Calculate L1 Distance
        # TODO x_data_test?
        distance = tf.reduce_sum(tf.abs(tf.add(input_, tf.negative(input_))), reduction_indices=1)
        # Calculate Manhattan Distance
        # distance = tf.reduce_sum(tf.abs(tf.subtract(input, tf.expand_dims(x_data_test, 1))), axis=2)

        # Prediction: Get min distance index (Nearest neighbor)
        pred = tf.argmin(distance, num_classes)
        return pred

    # ----------------------------------------------
    # compute convariance matrix of distance
    # ----------------------------------------------
    def compute_covariance(x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx, mx, vx

    # singular values of M_Mi give the variance:
    # use this to compute intrinsic dimensionality
    def compute_singular(x):
        return tf.linalg.svd(x, compute_uv=0)

    # use singular to compute intrinsic dimensionality of the data at this neighborhood.  The dimensionality is the
    # number of eigenvalues needed to sum to the total desired variance
    pred = find_nearest(input_)
    print(pred)

    cov_xx, mx, vx = compute_covariance(pred)
    sig = compute_singular(cov_xx)

    v = tf.constant(0.9)
    sig /= tf.reduce_sum(sig)
    S = tf.cumsum(sig)
    m_est = tf.searchsorted(S, v)

    f1 = tf.reduce_sum((v - S[m_est - 1]) / sig[m_est])
    f2 = v / sig[m_est]
    m_est = tf.case([(tf.less(tf.constant(0), m_est), f1)], default=f2)

    # Covariance matrix may be nearly singular:
    # add a diagonal correction to prevent numerical errors
    # correction is equal to the sum of the (d-m) unused variances (as in deRidder & Duin)
    r = tf.reduce_sum(sig)
    var_total += r

    # solve for weight
    w = tf.linalg.solve(mx, tf.ones(tf.shape(mx[0])))
    w /= tf.reduce_sum(w)

    # TODO to use this func between the tensorflow models, have to below params(row, nbrs)
    W[row, nbrs] = w

    # to find the null space, we need the bottom d+1, eigenvectors of (W-I).T*(W-I)
    # Compute this using the svd of (W-I):

    I = tf.identity(W.shape[0])
    U, sig, VT = tf.linalg.svd(W - I, full_amtrices=0)
    indices = tf.argsort(sig)[1:m + 1]

    return VT[indices, :]
