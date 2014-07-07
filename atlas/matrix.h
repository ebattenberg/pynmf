
//#include <cblas.h>
//#include <mkl_cblas.h>
#include<mkl.h>
#include <math.h>

#define EPS 1E-6



typedef struct{
    float* mat;
    int dim[2];
} matrix;

void read_matrix(matrix* A, char* file);
void write_matrix(matrix A, char* file);
void create_matrix(matrix* A, int rows, int cols, float value);
void destroy_matrix(matrix *A);
void print_matrix(matrix A);
void naive_matrix_multiply( matrix a, matrix b, matrix c); 
void matrix_transpose( matrix a, matrix b);
float matrix_difference_norm(matrix A, matrix B);
float matrix_difference_max(matrix A, matrix B);
float matrix_div(matrix A, matrix B);
float matrix_mean(matrix A);
inline void matrix_multiply( matrix a, matrix b, matrix c );
inline void matrix_multiply_AtB( matrix a, matrix b, matrix c );
inline void matrix_multiply_ABt( matrix a, matrix b, matrix c );
void element_multiply( matrix a, matrix b, matrix c);
void element_divide( matrix a, matrix b, matrix c);
void matrix_add_constant( matrix a, const float c);
void matrix_eps( matrix a);
void row_divide( matrix a, matrix b, matrix c);
void col_divide( matrix a, matrix b, matrix c);
void sum_rows( matrix a, matrix c);
void sum_cols( matrix a, matrix c);
