#include "python3.4/Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>

double *W, *fv, *fv2, *gv, *gv2;
long n_words, n_features, buffer[250000];
double n = 0.01; //default learning rate usually over-rode in Python.
static PyObject *qae_testweights(PyObject *self, PyObject *args);
double *start, *stop;

void checkW(int idx, char *s){
   if ( (W + idx) > stop || (W + idx) < start ) printf("%s OUT OF BOUNDS %p, start %p, stop %p\n",s, W + idx, start, stop); 
}

static PyObject *
qae_update_n(PyObject *self, PyObject *args)
{
    PyArg_ParseTuple(args, "d", &n);
    printf("n is now %lf.\n", n);
    return PyLong_FromLong(0);
}

void f(long len, long *idx){
	long i, j, offset;
	for (i=0; i < n_features; i++) fv[i] = 0.0;
	for (i=0;i< len; i++){
        offset = idx[i] * n_features;
		//printf("f word idx %ld\n", idx[ i ]);
		for (j=0; j< n_features; j++){
			fv[j] += W[ offset + j ] ;
		}
	}
	//for (i=0; i < n_features; i ++) printf("%lf ", fv[i]);
    //printf("\n");
}

void f2(long len, long *idx){
	long i, j, offset;
	for (i=0; i < n_features; i++) fv2[i] = 0.0;
	for (i=0;i< len; i++){
		//printf("Adding word %ld\n", i);
        offset = idx[i] * n_features;
		for (j=0; j< n_features; j++){
			fv2[j] += W[ offset + j ] ;
		}
	}
}

void g(long idx){
	long i;
    //printf("G idx is %ld\n", idx);
    idx *= n_features;
	for (i=0; i< n_features; i++){
		gv[i] = W[ idx + i ] ;
        //printf("WPi %p gv + i %p W[ ] %lf n_features %ld idx %ld i %ld gv[i] %lf\n",W + idx + i, gv + i,W[ idx + i ], n_features, idx, i, gv[i]);

	}
	//for (i=0; i < n_features; i ++) printf("%lf ", gv[i]);
    //printf("\n");
}

void g2(long len, long *idx){
	long i, j, offset;
	for (i=0; i < n_features; i++) gv2[i] = 0.0;
	for (i=0;i< len; i++){
		//printf("Adding word %ld\n", i);
        offset = idx[i] * n_features;
		for (j=0; j< n_features; j++){
			gv2[j] += W[ offset + j ] ;
		}
		for (j=0; j< n_features; j++){
			gv2[j] /= len;
		}
	}
}

static PyObject *
qae_s(PyObject *self, PyObject *args){
	long i;
	double dot = 0;

    Py_ssize_t len = PyTuple_Size(args);
    //printf("len is %ld\n", len);
    for (i=0;i<len;i++) buffer[i] = PyLong_AsLong( PyTuple_GetItem(args,i) );
    //for (i=0;i<len;i++) printf("buffer i %ld\n", buffer[i]);

    //qae_testweights();
	f(len - 1, buffer + 1);
	g(buffer[0]);
    //qae_testweights();
	for (i = 0; i < n_features; i ++) dot += fv[i] * gv[i];
    return PyFloat_FromDouble(dot);
}
 
static PyObject *
qae_sp(PyObject *self, PyObject *args){
	long i, len = 0, len2;
	double dot = 0;
    long *idx2;

    Py_ssize_t tuplesize = PyTuple_Size(args);
    for (i=0;i<tuplesize;i++) 
    {
        buffer[i] = PyLong_AsLong( PyTuple_GetItem(args,i) );
        if (buffer[i] < 0) len = i;
    }
    len2 = tuplesize - len - 1;
    idx2 = buffer + len + 1;

	f(len, buffer);
	f2(len2, idx2);
	for (i = 0; i < n_features; i ++) dot += fv[i] * fv2[i];
	//printf("\ndot %lf\n", dot);
    return PyFloat_FromDouble(dot);
}

static PyObject *
qae_s_multianswer(PyObject *self, PyObject *args){
    //Uses a feature averaging g.
	long i, len = 0, len2;
	double dot = 0;
    long *idx2;

    Py_ssize_t tuplesize = PyTuple_Size(args);
    for (i=0;i<tuplesize;i++) 
    {
        buffer[i] = PyLong_AsLong( PyTuple_GetItem(args,i) );
        if (buffer[i] < 0) len = i;
    }
    len2 = tuplesize - len - 1;
    idx2 = buffer + len + 1;

    //printf("len2: %ld, idx: %ld, len %ld, buffer: %ld\n", len2, idx2, len, buffer);
	f(len, buffer);
	g2(len2, idx2);
	for (i = 0; i < n_features; i ++) dot += fv[i] * gv2[i];
	//printf("\ndot %lf\n", dot);
    return PyFloat_FromDouble(dot);
}

void
constrain(long idx){
	long j;
	double norm = 0;
	for (j = 0; j < n_features; j++) {
		norm += W[ idx + j ] * W[ idx + j ];
		//printf("idx %ld j %ld W[idx+j] %lf", idx, j, W[idx + j]);
	}
	norm = sqrt(norm);
	if (norm > 1)	for (j = 0; j < n_features; j++){
		W[ idx + j ] /= norm;
        //checkW(idx + j, "constrain");
		//printf("yes idx %ld j %ld W[idx+j] %lf", idx, j, W[idx + j]);
		//printf(" %lf\n", norm);
	} 
}

//Create a function for decreasing the value of p_
static PyObject *
qae_update_dec(PyObject *self, PyObject *args){
	long i;

	long idx, idx2, offset, offset2;
    PyArg_ParseTuple(args, "ll", &idx, &idx2);

    //printf("called... idx %ld idx2 %ld\n", idx, idx2);
	idx *= n_features;
	idx2 *= n_features;
    //printf("W + idx %p, W + idx2 %p\n", W + idx, W + idx2);
    //printf("actual.. idx %ld idx2 %ld\n", idx, idx2);
	//original gradient does not work well
	//for (i = 0; i < n_features; i++) W[idx * n_features + i] += 0.1 * (W[idx * n_features + i] + W[idx2 * n_features + i]);
	//for (i = 0; i < n_features; i++) W[idx2 * n_features + i] += 0.1 * (W[idx * n_features + i] + W[idx2 * n_features + i]);
	for (i = 0; i < n_features; i++){
        //checkW(idx2 + i, "update_dec");
        //checkW(idx + i, "update_dec");
        offset  = idx  + i;
        offset2 = idx2 + i;
		if ( W[offset] >  0 && W[offset2] > 0 ) { 
			//printf("%lf =================dec\n", W[offset]);
			W[offset] -= n;
			//printf("%lf =================dec\n", W[offset]);
			W[offset2] -= n;
            continue;
		}
		if ( W[offset] > 0 && W[offset2] < 0 ) { 
			//printf("%lf +++++++++++++++++dec\n", W[offset]);
			W[offset] += n;
			//printf("%lf +++++++++++++++++dec\n", W[offset]);
			W[offset2] -= n;
            continue;
		}
		if ( W[offset] < 0 && W[offset2] > 0 ) { 
			//printf("%lf ^^^^^^^^^^^^^^^^dec\n", W[offset]);
			W[offset] -= n;
			//printf("%lf ^^^^^^^^^^^^^^^^dec\n", W[offset]);
			W[offset2] += n;
            continue;
		}
        //ELSE
        //printf("%lf ***************dec\n", W[offset]);
        W[offset] += n;
        //printf("%lf ***************dec\n", W[offset]);
        W[offset2] += n;
		//printf("%lf ", W[idx2 + i]);
	}
	constrain(idx);
	constrain(idx2);
	//printf("\n");
    return PyLong_FromLong(0);
}

//updates predicate once for each word.
static PyObject *
//qae_update(long idx, long idx2){
qae_update(PyObject *self, PyObject *args){
	long i;

	long idx, idx2, offset, offset2;
    PyArg_ParseTuple(args, "ll", &idx, &idx2);

    //printf("called... idx %ld idx2 %ld\n", idx, idx2);
	idx *= n_features;
	idx2 *= n_features;
    //printf("W + idx %p, W + idx2 %p\n", W + idx, W + idx2);
    //printf("actual... idx %ld idx2 %ld\n", idx, idx2);
	//original gradient does not work well
	//for (i = 0; i < n_features; i++) W[idx * n_features + i] += 0.1 * (W[idx * n_features + i] + W[idx2 * n_features + i]);
	//for (i = 0; i < n_features; i++) W[idx2 * n_features + i] += 0.1 * (W[idx * n_features + i] + W[idx2 * n_features + i]);
	for (i = 0; i < n_features; i++){
        //checkW(idx2 + i, "update");
        //checkW(idx + i, "update");
        offset  = idx  + i;
        offset2 = idx2 + i;
		if ( W[offset] >  0 && W[offset2] > 0 ) { 
			//printf("%lf =================\n", W[offset]);
			W[offset] += n;
			//printf("%lf =================\n", W[offset]);
			W[offset2] += n;
            continue;
		}
		if ( W[offset] > 0 && W[offset2] < 0 ) { 
			//printf("%lf +++++++++++++++++\n", W[offset]);
			W[offset] -= n;
			//printf("%lf +++++++++++++++++\n", W[offset]);
			W[offset2] += n;
            continue;
		}
		if ( W[offset] < 0 && W[offset2] > 0 ) { 
			//printf("%lf ^^^^^^^^^^^^^^^^\n", W[offset]);
			W[offset] += n;
			//printf("%lf ^^^^^^^^^^^^^^^^\n", W[offset]);
			W[offset2] -= n;
            continue;
		}
        //ELSE
		//both negative
        //printf("%lf ***************\n", W[offset]);
        W[offset] -= n;
        //printf("%lf ***************\n", W[offset]);
        W[offset2] -= n;
	//	printf("%lf ", W[idx2 + i]);
	}
	constrain(idx);
	constrain(idx2);
	//printf("\n");
    return PyLong_FromLong(0);
}
  
static PyObject *
qae_initialize(PyObject *self, PyObject *args){

    PyArg_ParseTuple(args, "ll", &n_words, &n_features);

	//W = malloc(n_words * n_features * sizeof(double));
    int fd = shm_open("W-Weights", O_CREAT | O_RDWR, 0666);
    if (fd == -1){
        perror("shm open");
        exit(1);
    }
    int r = ftruncate(fd, sizeof(double) * n_words * n_features);
    if (r != 0)
    {
          printf("ftruncate");
          exit(1);
    }

    W = (double*) mmap(NULL, sizeof(double) * n_words * n_features, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (W == MAP_FAILED) {
        perror("mmap failed.");
        exit(1);
    }
    printf("mmap suceeded.\n");
	fv = (double*) malloc(n_features * sizeof(double));
	fv2 = (double*) malloc(n_features * sizeof(double));
	gv = (double*) malloc(n_features * sizeof(double));
	gv2 = (double*) malloc(n_features * sizeof(double));

	srand( (unsigned)time( NULL ) );

	long i;
	long len = n_words * n_features;

  //  printf("about to loop.\n");
	for (i = 0; i < len; i ++) {
//        printf("W[i] is %lf at index %ld\n", W[i], i);
		W[i] = (double)(((double)rand()) / ((double)RAND_MAX)) - 0.5;
 //       printf("W[i] is %lf at index %ld\n", W[i], i);
		//if (W[i] > 1 || W[i] < -1) printf("\nWEIGHT SET AT %lf AT INDEX %ld\n", W[i], i);
	}
   // printf("w initialized\n");

    //start = W;
    //stop = W + (n_words * n_features);
    //printf("start %p, stop %p\n", W, W + (n_words * n_features) );
    return PyLong_FromLong(0);
}
  
static PyObject *
qae_saveweights(PyObject *self, PyObject *args){ 
	FILE *fp = fopen("weights.bin", "wb");
    if ( NULL == fp ){
        printf("failed file open.\n");
        exit(1);
    }
	printf("Saving n_words %ld  n_features %ld\n", n_words, n_features);
    if (0 == fwrite(&n_words, 1, sizeof(long), fp) || \
        0 == fwrite(&n_features, 1, sizeof(long), fp) || \
        0 == fwrite(W, 1, n_words * n_features * sizeof(double), fp)) {
        printf("fwrite failed.\n");
        exit(1);
    }
	fclose(fp);
    return PyLong_FromLong(0);
}

static PyObject *
qae_loadweights(PyObject *self, PyObject *args){ 
	FILE *fp = fopen("weights.bin", "rb");
    if ( NULL == fp ){
        printf("failed file open.\n");
        exit(1);
    }
    if ( 0 == fread(&n_words, 1, sizeof(long), fp) ){
        printf("fread failed\n");
        exit(1);
    }
    if ( 0 == fread(&n_features, 1, sizeof(long), fp) ){
        printf("fread failed\n");
        exit(1);
    }
    int fd = shm_open("W-Weights", O_CREAT | O_RDWR, 0666);
    if (fd == -1){
        perror("shm open");
        exit(1);
    }
    int r = ftruncate(fd, sizeof(double) * n_words * n_features);
    if (r != 0)
    {
          printf("ftruncate");
          exit(1);
    }
    W = (double*) mmap(NULL, sizeof(double) * n_words * n_features, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (W == MAP_FAILED) {
        perror("mmap failed.");
        exit(1);
    }
    if ( 0 == fread(W, 1, n_words * n_features * sizeof(double), fp) ){
        printf("fread failed\n");
        exit(1);
    }
	fclose(fp);

	fv = (double*) malloc(n_features * sizeof(double));
	fv2 = (double*) malloc(n_features * sizeof(double));
	gv = (double*) malloc(n_features * sizeof(double));
	gv2 = (double*) malloc(n_features * sizeof(double));

    return PyLong_FromLong(0);
}

static PyObject *
qae_displayweights(PyObject *self, PyObject *args){
	long i;
	for (i = 0; i < n_words * n_features;i++) printf("%lf -\n", W[i]);
    printf("%p", W);
	printf("Displayed Weights!\n");
	fflush(stdout);
    return PyLong_FromLong(0);
}

static PyObject *
qae_testweights(PyObject *self, PyObject *args){
	long i;
    for (i = 0; i < n_words * n_features;i++) if (W[i] > 2 || W[i] < -2 || W[i] == 0) printf("\nIT SHOULDN'T BE THIS BIG %lf -\n", W[i]);
	fflush(stdout);
    return PyLong_FromLong(0);
}

static PyMethodDef QaeMethods[] = 
{
    {"update_n", qae_update_n, METH_VARARGS, ""},
    {"s", qae_s, METH_VARARGS, ""},
    {"sp", qae_sp, METH_VARARGS, ""},
    {"s_multianswer", qae_s_multianswer, METH_VARARGS, ""},
    {"update_dec", qae_update_dec, METH_VARARGS, ""},
    {"update", qae_update, METH_VARARGS, ""},
    {"initialize", qae_initialize, METH_VARARGS, ""},
    {"saveweights", qae_saveweights, METH_VARARGS, ""},
    {"loadweights", qae_loadweights, METH_VARARGS, ""},
    {"displayweights", qae_displayweights, METH_VARARGS, ""},
    {"testweights", qae_testweights, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef qaemodule = {
    PyModuleDef_HEAD_INIT,
        "qae",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
        QaeMethods
};

PyMODINIT_FUNC
PyInit_qae(void)
{
        return PyModule_Create(&qaemodule);
}
