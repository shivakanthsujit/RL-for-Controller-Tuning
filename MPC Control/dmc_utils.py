def dmccalc(Sp, Kmat, sn, delup, d, r, u, k, n):
    """
    for use with dmcsim.m
    b.w. bequette
    2 oct 00
    calculate the optimum control move

    """
    #  first, calculate uold = u(k-n+1)...u(k-n+p)
    m, p = Kmat.shape
    uold = np.zeros((p, 1))

    for i in range(p):
        if k - n + i + 1 >= 0:
            uold[i] = u[k - n + i + 1]
        else:
            uold[i] = 0
    dvec = d * np.ones((p, 1))
    rvec = r * np.ones((p, 1))
    y_free = np.matmul(Sp, delup) + sn * uold + dvec
    e_free = rvec - y_free
    delu = np.matmul(Kmat[0], e_free)
    return delu


def smatgen(s, p, m, n, w):
    """
    b.w. bequette
    28 Sept 00, revised 2 Oct 00
    generates dynamic matrix and feedback gain matrix
    assumes s = step response column vector
    Sf = Dynamic Matrix for future control moves (forced)
    Sp = Matrix for past control moves (free)
    Kmat = DMC feedback gain matrix
    s = step response coefficient vector
    p = prediction horizon
    m = control horizon
    n = model horizon
    w = weight on control input
    """
    Sf = np.zeros((p, m))
    Sp = np.zeros((p, n - 2))
    #  first, find the dynamic matrix
    for j in range(m):
        Sf[:, j] = np.append(np.zeros((j, 1)), s[0 : p - j])

    #  now, find the matrix for past moves
    for i in range(p):
        Sp[i, :] = np.append(np.transpose(s[i + 1 : n - 1]), np.zeros((1, i)))

    #  find the feedback gain matrix, Kmat
    Kmat = np.matmul(
        inv(np.matmul(np.transpose(Sf), Sf) + w * np.eye(m)), np.transpose(Sf)
    )
    return Sf, Sp, Kmat
