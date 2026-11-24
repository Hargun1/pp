#include <stdio.h>
#include <omp.h>

int x = 10;
#pragma omp threadprivate(x)

int main(void) {
    printf("Master thread initial x = %d\n", x);

    #pragma omp parallel num_threads(4) copyin(x)
    {
        int id = omp_get_thread_num();
        printf("Thread %d initial x = %d\n", id, x);

        x = x + id;
        printf("Thread %d updated x = %d\n", id, x);
    }

    printf("Master thread final x = %d\n", x);
    return 0;
}
