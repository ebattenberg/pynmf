
typedef struct{
    float* mat;
    float* mat_d;
    int dim[2];
} matrix;

void element_add(matrix a, matrix b, matrix c);
void copy_matrix_to_device(matrix* A);
void copy_matrix_from_device(matrix* A);




