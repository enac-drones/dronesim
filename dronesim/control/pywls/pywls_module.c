#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <string.h>

#include "wls_alloc.h"

/* --------------------------- Helper utilities --------------------------- */

static PyArrayObject *
require_float32_array(PyObject *obj, int ndim, const char *name)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (arr == NULL) {
        return NULL;  /* NumPy already set an exception */
    }

    if (PyArray_NDIM(arr) != ndim) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a %d-D NumPy array",
                     name, ndim);
        Py_DECREF(arr);
        return NULL;
    }

    return arr;
}

static int
check_vector_len(PyArrayObject *arr, npy_intp expected_len, const char *name)
{
    if (PyArray_NDIM(arr) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1-D", name);
        return -1;
    }

    if (PyArray_DIM(arr, 0) != expected_len) {
        PyErr_Format(PyExc_ValueError,
                     "%s has length %lld, expected %lld",
                     name,
                     (long long)PyArray_DIM(arr, 0),
                     (long long)expected_len);
        return -1;
    }

    return 0;
}

static int
check_matrix_shape(PyArrayObject *arr, npy_intp rows, npy_intp cols, const char *name)
{
    if (PyArray_NDIM(arr) != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be 2-D", name);
        return -1;
    }

    if (PyArray_DIM(arr, 0) != rows || PyArray_DIM(arr, 1) != cols) {
        PyErr_Format(PyExc_ValueError,
                     "%s has shape (%lld, %lld), expected (%lld, %lld)",
                     name,
                     (long long)PyArray_DIM(arr, 0),
                     (long long)PyArray_DIM(arr, 1),
                     (long long)rows,
                     (long long)cols);
        return -1;
    }

    return 0;
}

/* ----------------------------- Wrapped call ----------------------------- */

PyDoc_STRVAR(pywls_wls_alloc_doc,
"wls_alloc(B, v, u_min, u_max, u_guess=None, W_init=None, Wv=None, Wu=None, "
"u_pref=None, gamma_sq=100000.0, imax=100)\n"
"--\n"
"\n"
"Wrap the C function wls_alloc() from wls_alloc.c.\n"
"\n"
"Parameters\n"
"----------\n"
"B : ndarray, shape (nv, nu), float32-compatible\n"
"    Control effectiveness matrix.\n"
"v : ndarray, shape (nv,), float32-compatible\n"
"    Control objective vector.\n"
"u_min, u_max : ndarray, shape (nu,), float32-compatible\n"
"    Lower/upper actuator limits.\n"
"u_guess : ndarray, optional, shape (nu,)\n"
"    Initial actuator guess.\n"
"W_init : ndarray, optional, shape (nu,)\n"
"    Initial working set.\n"
"Wv : ndarray, optional, shape (nv,)\n"
"    Objective weights. Defaults to ones.\n"
"Wu : ndarray, optional, shape (nu,)\n"
"    Control weights. Defaults to ones.\n"
"u_pref : ndarray, optional, shape (nu,)\n"
"    Preferred actuator vector. Defaults to zeros.\n"
"gamma_sq : float, optional\n"
"    Weighting factor.\n"
"imax : int, optional\n"
"    Maximum number of iterations.\n"
"\n"
"Returns\n"
"-------\n"
"(u, iter) : tuple\n"
"    u is a float32 NumPy array of shape (nu,), iter is the iteration count.\n");

static PyObject *
pywls_wls_alloc(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;

    PyObject *B_obj = NULL;
    PyObject *v_obj = NULL;
    PyObject *u_min_obj = NULL;
    PyObject *u_max_obj = NULL;
    PyObject *u_guess_obj = Py_None;
    PyObject *W_init_obj = Py_None;
    PyObject *Wv_obj = Py_None;
    PyObject *Wu_obj = Py_None;
    PyObject *u_pref_obj = Py_None;
    float gamma_sq = 100000.0f;
    int imax = 100;

    static char *kwlist[] = {
        "B", "v", "u_min", "u_max",
        "u_guess", "W_init", "Wv", "Wu", "u_pref",
        "gamma_sq", "imax",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs,
            "OOOO|OOOOOfi:wls_alloc",
            kwlist,
            &B_obj, &v_obj, &u_min_obj, &u_max_obj,
            &u_guess_obj, &W_init_obj, &Wv_obj, &Wu_obj, &u_pref_obj,
            &gamma_sq, &imax)) {
        return NULL;
    }

    PyArrayObject *B_arr = NULL;
    PyArrayObject *v_arr = NULL;
    PyArrayObject *u_min_arr = NULL;
    PyArrayObject *u_max_arr = NULL;
    PyArrayObject *u_guess_arr = NULL;
    PyArrayObject *W_init_arr = NULL;
    PyArrayObject *Wv_arr = NULL;
    PyArrayObject *Wu_arr = NULL;
    PyArrayObject *u_pref_arr = NULL;
    PyArrayObject *u_out_arr = NULL;

    float **B_rows = NULL;
    PyObject *result = NULL;

    npy_intp nu = 0;
    npy_intp nv = 0;

    struct WLS_t state;
    memset(&state, 0, sizeof(state));

    /* Required arrays */
    v_arr = require_float32_array(v_obj, 1, "v");
    if (v_arr == NULL) goto fail;
    nv = PyArray_DIM(v_arr, 0);

    u_min_arr = require_float32_array(u_min_obj, 1, "u_min");
    if (u_min_arr == NULL) goto fail;
    nu = PyArray_DIM(u_min_arr, 0);

    u_max_arr = require_float32_array(u_max_obj, 1, "u_max");
    if (u_max_arr == NULL) goto fail;
    if (check_vector_len(u_max_arr, nu, "u_max") < 0) goto fail;

    B_arr = require_float32_array(B_obj, 2, "B");
    if (B_arr == NULL) goto fail;
    if (check_matrix_shape(B_arr, nv, nu, "B") < 0) goto fail;

    /* Enforce compile-time limits from wls_alloc.h */
    if (nu > WLS_N_U_MAX) {
        PyErr_Format(PyExc_ValueError,
                     "nu=%lld exceeds compile-time WLS_N_U_MAX=%d; "
                     "rebuild the extension with a larger WLS_N_U_MAX",
                     (long long)nu, WLS_N_U_MAX);
        goto fail;
    }
    if (nv > WLS_N_V_MAX) {
        PyErr_Format(PyExc_ValueError,
                     "nv=%lld exceeds compile-time WLS_N_V_MAX=%d; "
                     "rebuild the extension with a larger WLS_N_V_MAX",
                     (long long)nv, WLS_N_V_MAX);
        goto fail;
    }

    /* Optional arrays */
    if (u_guess_obj != Py_None) {
        u_guess_arr = require_float32_array(u_guess_obj, 1, "u_guess");
        if (u_guess_arr == NULL) goto fail;
        if (check_vector_len(u_guess_arr, nu, "u_guess") < 0) goto fail;
    }

    if (W_init_obj != Py_None) {
        W_init_arr = require_float32_array(W_init_obj, 1, "W_init");
        if (W_init_arr == NULL) goto fail;
        if (check_vector_len(W_init_arr, nu, "W_init") < 0) goto fail;
    }

    if (Wv_obj != Py_None) {
        Wv_arr = require_float32_array(Wv_obj, 1, "Wv");
        if (Wv_arr == NULL) goto fail;
        if (check_vector_len(Wv_arr, nv, "Wv") < 0) goto fail;
    }

    if (Wu_obj != Py_None) {
        Wu_arr = require_float32_array(Wu_obj, 1, "Wu");
        if (Wu_arr == NULL) goto fail;
        if (check_vector_len(Wu_arr, nu, "Wu") < 0) goto fail;
    }

    if (u_pref_obj != Py_None) {
        u_pref_arr = require_float32_array(u_pref_obj, 1, "u_pref");
        if (u_pref_arr == NULL) goto fail;
        if (check_vector_len(u_pref_arr, nu, "u_pref") < 0) goto fail;
    }

    /* Build float** row pointers for B */
    B_rows = (float **)PyMem_Malloc((size_t)nv * sizeof(float *));
    if (B_rows == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    {
        float *B_data = (float *)PyArray_DATA(B_arr);
        for (npy_intp i = 0; i < nv; ++i) {
            B_rows[i] = B_data + i * nu;
        }
    }

    /* Fill WLS_t */
    state.nu = (int)nu;
    state.nv = (int)nv;
    state.gamma_sq = gamma_sq;
    state.iter = 0;

    memcpy(state.v,     PyArray_DATA(v_arr),     (size_t)nv * sizeof(float));
    memcpy(state.u_min, PyArray_DATA(u_min_arr), (size_t)nu * sizeof(float));
    memcpy(state.u_max, PyArray_DATA(u_max_arr), (size_t)nu * sizeof(float));

    if (Wv_arr != NULL) {
        memcpy(state.Wv, PyArray_DATA(Wv_arr), (size_t)nv * sizeof(float));
    } else {
        for (npy_intp i = 0; i < nv; ++i) state.Wv[i] = 1.0f;
    }

    if (Wu_arr != NULL) {
        memcpy(state.Wu, PyArray_DATA(Wu_arr), (size_t)nu * sizeof(float));
    } else {
        for (npy_intp i = 0; i < nu; ++i) state.Wu[i] = 1.0f;
    }

    if (u_pref_arr != NULL) {
        memcpy(state.u_pref, PyArray_DATA(u_pref_arr), (size_t)nu * sizeof(float));
    } else {
        for (npy_intp i = 0; i < nu; ++i) state.u_pref[i] = 0.0f;
    }

    /*
     * wls_alloc() initializes state.u from u_guess if provided, otherwise
     * from the midpoint of [u_min, u_max].
     */
    Py_BEGIN_ALLOW_THREADS
    wls_alloc(
        &state,
        B_rows,
        (u_guess_arr != NULL) ? (float *)PyArray_DATA(u_guess_arr) : NULL,
        (W_init_arr != NULL) ? (float *)PyArray_DATA(W_init_arr) : NULL,
        imax
    );
    Py_END_ALLOW_THREADS

    {
        npy_intp dims[1] = { nu };
        u_out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
        if (u_out_arr == NULL) goto fail;
        memcpy(PyArray_DATA(u_out_arr), state.u, (size_t)nu * sizeof(float));
    }

    result = Py_BuildValue("Ni", (PyObject *)u_out_arr, state.iter);
    u_out_arr = NULL;  /* stolen by "N" */

fail:
    Py_XDECREF(B_arr);
    Py_XDECREF(v_arr);
    Py_XDECREF(u_min_arr);
    Py_XDECREF(u_max_arr);
    Py_XDECREF(u_guess_arr);
    Py_XDECREF(W_init_arr);
    Py_XDECREF(Wv_arr);
    Py_XDECREF(Wu_arr);
    Py_XDECREF(u_pref_arr);
    Py_XDECREF(u_out_arr);

    if (B_rows != NULL) PyMem_Free(B_rows);

    return result;
}

/* ------------------------------ Module table ---------------------------- */

static PyMethodDef pywls_methods[] = {
    {
        "wls_alloc",
        (PyCFunction)pywls_wls_alloc,
        METH_VARARGS | METH_KEYWORDS,
        pywls_wls_alloc_doc
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef pywls_module = {
    PyModuleDef_HEAD_INIT,
    "pywls",
    "CPython/NumPy wrapper for wls_alloc.c",
    -1,
    pywls_methods
};

PyMODINIT_FUNC
PyInit_pywls(void)
{
    PyObject *m = PyModule_Create(&pywls_module);
    if (m == NULL) {
        return NULL;
    }

    import_array();
    return m;
}

