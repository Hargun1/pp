#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("OpenMP directives demo\n");

    #pragma omp parallel num_threads(4)
    {
        printf("Parallel: thread %d\n", omp_get_thread_num());

        #pragma omp single
        {
            printf("Single block executed by thread %d\n",
                   omp_get_thread_num());
        }

        #pragma omp for
        for (int i = 1; i <= 8; i++) {
            printf("For: thread %d handles i=%d\n",
                   omp_get_thread_num(), i);
        }

        #pragma omp sections
        {
            #pragma omp section
            {
                printf("Section 1: addition = %d\n", 10 + 20);
            }

            #pragma omp section
            {
                printf("Section 2: subtraction = %d\n", 20 - 10);
            }

            #pragma omp section
            {
                printf("Section 3: multiplication = %d\n", 10 * 20);
            }
        }
    }

    return 0;
}
