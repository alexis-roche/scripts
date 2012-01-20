#include "mrf.h"

#include <math.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define inline __inline
#endif


/* Numpy import */
void mrf_import_array(void) { 
  import_array(); 
  return;
}

/* Encode neighborhood systems using static arrays */
int ngb6 [] = {1,0,0,
	       -1,0,0,
	       0,1,0,
	       0,-1,0,
	       0,0,1,
	       0,0,-1}; 

int ngb26 [] = {1,0,0,
		-1,0,0,
		0,1,0,
		0,-1,0,
		1,1,0,
		-1,-1,0,
		1,-1,0,
		-1,1,0, 
		1,0,1,
		-1,0,1,
		0,1,1,
		0,-1,1, 
		1,1,1,
		-1,-1,1,
		1,-1,1,
		-1,1,1, 
		1,0,-1,
		-1,0,-1,
		0,1,-1,
		0,-1,-1, 
		1,1,-1,
		-1,-1,-1,
		1,-1,-1,
		-1,1,-1, 
		0,0,1,
		0,0,-1}; 



static int* _select_neighborhood_system(int ngb_size) {
  if (ngb_size == 6) { 
    fprintf(stderr, "6-neighborhood system\n"); 
    return ngb6;
  }
  else if (ngb_size == 26) {
    fprintf(stderr, "26-neighborhood system\n"); 
    return ngb26;
  }
  else {
    fprintf(stderr, "Unknown neighborhood system\n");
    return NULL; 
  }
}



/* Compute the (negated) expected interaction energy of a voxel with
   some neighbor */
static inline void _get_message(double* res, int K, size_t pos, 
				const double* ppm_data, const double* aux)
{
  double *buf = res, *buf_ppm = (double*)ppm_data + pos;
  int k;
  
  for (k=0; k<K; k++, buf++, buf_ppm++)
    *buf += *buf_ppm;

  return;
}

static inline void _finalize_inbox(double* res, int K, const double* aux) 
{
  int k; 
  double* buf;
  double aux0 = aux[0];

  for (k=0, buf=res; k<K; k++, buf++) 
    *buf = exp(aux0 * (*buf));

  return; 
}

static inline void _initialize_inbox(double* res, int K)
{ 
  memset ((void*)res, 0, K*sizeof(double));
  return; 
}

static inline void _get_message_icm(double* res, int K, size_t pos,  
				    const double* ppm_data, const double* aux)
{
  int k, kmax = -1;
  double max = 0, tmp;
  double *buf_ppm = (double*)ppm_data + pos;

  for (k=0; k<K; k++, buf_ppm++) {
    tmp = *buf_ppm;
    if (tmp>max) {
      kmax = k;
      max = tmp;
    }
  }
  if (kmax >= 0)
    res[kmax] += 1;

  return;
}

static inline void _get_message_bp(double* res, int K, size_t pos, 
				   const double* ppm_data, const double* aux)
{
  double *buf = res, *buf_ppm = (double*)ppm_data + pos;
  int k;
  double aux0 = aux[0]; 

  for (k=0; k<K; k++, buf++, buf_ppm++) 
    *buf *= aux0 * (*buf_ppm) + 1; 
  
  return;
}

static inline void _initialize_inbox_bp(double* res, int K)
{ 
  double *buf = res; 
  int k; 

  for (k=0; k<K; k++, buf++)
    *buf = 1.0; 

  /* memset ((void*)res, 1, K*sizeof(double));*/
  return; 
}



/*
  Compute the incoming messages at a given voxel as a function of the
  class label, and aggregate them across neighbors.
  
  The vode_fn argument is a pointer to the function that actually
  computes the expected interaction energy with a particular neighbor.

  ppm assumed contiguous double (X, Y, Z, K) 
  res assumed preallocated with size >= K 

*/

static void _ngb_compound_messages(double* res,
				   const PyArrayObject* ppm,
				   int x,
				   int y, 
				   int z,
				   void* initialize_inbox,
				   void* get_message,
				   void* finalize_inbox,
				   const double* aux,
				   const int* ngb,
				   int ngb_size)			
{
  int j = 0, xn, yn, zn, K = ppm->dimensions[3]; 
  const int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  size_t u3 = K; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  size_t pos; 
  void (*_initialize_inbox)(double*,int) = initialize_inbox;
  void (*_get_message)(double*,int,size_t,const double*,const double*) = get_message;
  void (*_finalize_inbox)(double*,int,const double*) = finalize_inbox;

  /*  Re-initialize output array */
  _initialize_inbox(res, K); 

  /* Loop over neighbors */
  buf_ngb = ngb;
  while (j < ngb_size) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*u3;
    _get_message(res, K, pos, ppm_data, aux);
    j ++; 
  }

  /* Finalize total message computation */
  if (_finalize_inbox != NULL) 
    _finalize_inbox(res, K, aux); 
  
  return; 
}




/*
  ppm assumed contiguous double (X, Y, Z, K) 
  ref assumed contiguous double (NPTS, K)
  XYZ assumed contiguous usigned int (NPTS, 3)
*/

#define TINY 1e-300

void ve_step(PyArrayObject* ppm, 
	     const PyArrayObject* ref,
	     const PyArrayObject* XYZ,
	     int ngb_size, 
	     double beta,
	     int copy,
	     int scheme)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  const double* ref_data = (double*)ref->data;
  size_t v1 = ref->dimensions[1];
  int* xyz; 
  int* ngb;
  void (*initialize_inbox)(double*,int);
  void (*get_message)(double*,int,size_t,const double*,const double*);
  void (*finalize_inbox)(double*,int,const double*);
  size_t S; 
  double* aux;

  /* Dimensions */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
  S = PyArray_SIZE(ppm);

  /* Copy or not copy */
  if (copy) {
    ppm_data = (double*)calloc(S, sizeof(double));
    if (ppm_data==NULL) {
      fprintf(stderr, "Cannot allocate ppm copy\n"); 
      return; 
    }
    memcpy((void*)ppm_data, (void*)ppm->data, S*sizeof(double));
  }
  else
    ppm_data = (double*)ppm->data;
  
  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Select message passing scheme: mean-field, ICM or
     belief-propagation */
  switch (scheme) {
  case 0: 
    {
      initialize_inbox = &_initialize_inbox;
      get_message = &_get_message;
      finalize_inbox = &_finalize_inbox;
      aux = (double*)calloc(1, sizeof(double));   
      aux[0] = beta; 
    }
    break; 
  case 1: 
    {
      initialize_inbox = &_initialize_inbox;
      get_message = &_get_message_icm;
      finalize_inbox = &_finalize_inbox;
      aux = (double*)calloc(1, sizeof(double));
      aux[0] = beta; 
    }
    break; 
  case 2: 
    {
      initialize_inbox = &_initialize_inbox_bp;
      get_message = &_get_message_bp;    
      finalize_inbox = NULL; 
      aux = (double*)calloc(1, sizeof(double));
      aux[0] = exp(beta) - 1; 
      if (aux[0] < 0) 
	aux[0] = 0; 
    }
    break; 
  default: 
    {
      fprintf(stderr, "Unknown message-passing scheme\n"); 
      return; 
    }
    break; 
  }

  /* Allocate auxiliary vectors */
  p = (double*)calloc(K, sizeof(double)); 

  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);

  while(iter->index < iter->size) {

    /* Compute the average ppm in the neighborhood */
    xyz = PyArray_ITER_DATA(iter); 
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_compound_messages(p, ppm, x, y, z, 
			   (void*)initialize_inbox, 
			   (void*)get_message, 
			   (void*)finalize_inbox, 
			   aux, (const int*)ngb, ngb_size); 
    
    /* Multiply with reference and compute normalization constant */
    psum = 0.0; 
    for (k=0, kk=(iter->index)*v1, buf=p; k<K; k++, kk++, buf++) {
      tmp = (*buf) * ref_data[kk];
      psum += tmp; 
      *buf = tmp; 
    }
    
    /* Normalize to unitary sum */
    kk = x*u1 + y*u2 + z*u3; 
    if (psum > TINY) 
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = *buf/psum; 
    else
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = (*buf+TINY/(double)K)/(psum+TINY); 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 
  
  }

  /* If applicable, copy back the auxiliary ppm array into the input */ 
  if (copy) {    
    memcpy((void*)ppm->data, (void*)ppm_data, S*sizeof(double));
    free(ppm_data);
  }

  /* Free memory */ 
  free(p);
  if (aux != NULL) 
    free(aux); 
  Py_XDECREF(iter);

  return; 
}



double interaction_energy(PyArrayObject* ppm, 
			  const PyArrayObject* XYZ,
			  int ngb_size)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double res = 0.0, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  int* xyz; 
  int* ngb;

  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Dimensions */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
  ppm_data = (double*)ppm->data;

  /* Allocate auxiliary vector */
  p = (double*)calloc(K, sizeof(double)); 
  
  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);
  while(iter->index < iter->size) {
    
    /* Compute the average ppm in the neighborhood */ 
    xyz = PyArray_ITER_DATA(iter); 
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_compound_messages(p, ppm, x, y, z, &_initialize_inbox, 
			   &_get_message, NULL, NULL,
			   (const int*)ngb, ngb_size); 
    
    /* Calculate the dot product <q,p> where q is the local
       posterior */
    tmp = 0.0; 
    kk = x*u1 + y*u2 + z*u3; 
    for (k=0, buf=p; k<K; k++, kk++, buf++)
      tmp += ppm_data[kk]*(*buf);

    /* Update overall energy */ 
    res += tmp; 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 
  }

  /* Free memory */ 
  free(p);
  Py_XDECREF(iter);

  return res; 
}



static void _ngb_integrate(double* res,
			   const PyArrayObject* ppm,
			   int x,
			   int y, 
			   int z,
			   const double* U, 
			   double beta, 
			   const int* ngb,
			   int ngb_size)			
{
  int j = 0, xn, yn, zn, k, kk, K = ppm->dimensions[3]; 
  const int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  double *buf, *buf_ppm, *q, *buf_U;
  size_t u3 = K; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  size_t pos; 

  /*  Re-initialize output array */
  memset ((void*)res, 0, K*sizeof(double));

  /* Loop over neighbors */
  buf_ngb = ngb; 
  while (j < ngb_size) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*u3;

    /* Compute U*q */
    buf_ppm = (double*)ppm_data + pos;
    for (k=0, buf=res, buf_U=(double*)U; k<K; k++, buf++)
      for (kk=0, q=buf_ppm; kk<K; kk++, q++, buf_U++)
	*buf += *buf_U * *q;

    j ++; 
  }

  /* Finalize total message computation */
  for (k=0, buf=res; k<K; k++, buf++) 
    *buf = exp(-beta * (*buf));

  return; 
}


void gen_ve_step(PyArrayObject* ppm, 
		 const PyArrayObject* ref,
		 const PyArrayObject* XYZ, 
		 const PyArrayObject* U,
		 int ngb_size,
		 double beta)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  const double* ref_data = (double*)ref->data;
  const double* U_data = (double*)U->data;
  size_t v1 = ref->dimensions[1];
  int* xyz; 
  int* ngb;

  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Number of classes */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);

  /* Pointer to the data array */
  ppm_data = (double*)ppm->data;
  
  /* Allocate auxiliary vectors */
  p = (double*)calloc(K, sizeof(double)); 

  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);

  while(iter->index < iter->size) {

    /* Compute the average ppm in the neighborhood */
    xyz = PyArray_ITER_DATA(iter); 
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_integrate(p, ppm, x, y, z, U_data, beta, (const int*)ngb, ngb_size);
    
    /* Multiply with reference and compute normalization constant */
    psum = 0.0; 
    for (k=0, kk=(iter->index)*v1, buf=p; k<K; k++, kk++, buf++) {
      tmp = (*buf) * ref_data[kk];
      psum += tmp; 
      *buf = tmp; 
    }
    
    /* Normalize to unitary sum */
    kk = x*u1 + y*u2 + z*u3; 
    if (psum > TINY) 
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = *buf/psum; 
    else
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = (*buf+TINY/(double)K)/(psum+TINY); 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 
  
  }

  /* Free memory */ 
  free(p);
  Py_XDECREF(iter);

  return; 
}


/* Given a mask, compute list of edges corresponding to a given
   neighborhood system */

void make_edges(PyArrayObject* edges, 
		const PyArrayObject* mask,
		int ngb_size)
{
  return;
}
