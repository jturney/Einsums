
    /*
     * Calling convention:
     * Integer and pointer arguments: rdi, rsi, rdx, rcx, r8, r9
     * Floating point: xmm0-7
     * The rest: stack
     */
    .section .note.GNU-stack,"",@progbits
    .text
    .global ddirprod_kernel_avx2
    .type ddirprod_kernel_avx2,@function
    .arch .avx2
    /**
     * Direct product kernel for when all strides are 1.
     */
ddirprod_kernel_avx2:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa    %rsp, 8
    movq    %rsp, %rbp
    .cfi_def_cfa    %rbp, 8
    subq    $160, %rsp

    // Save the arguments
    movq    %rdi, -8(%rbp)
    movq    %rsi, -16(%rbp)
    movq    %rdx, -24(%rbp)
    movq    %rcx, -32(%rbp)

    // Save the other registers.
    movq    %rax, -40(%rbp)
    movq    %rbx, -48(%rbp)

    // Save the floating point registers.
    movupd  %xmm0, -64(%rbp)

    // Save extra floating point registers.
    vmovupd %ymm1, -96(%rbp)
    vmovupd %ymm2, -128(%rbp)
    vmovupd %ymm3, -160(%rbp)

    // Save rbx and move the output pointer to it.
    movq    %rcx, %rbx

    // Calculate the excess elements after the vector stuff.
    movq    %rdi, %rax
    shrq    $2, %rax
    movq    %rdi, %rcx
    andq    $3, %rcx

    // Save the number of excess.
    pushq   %rcx

    // Broadcast the prefactor.
    vbroadcastsd    %xmm0, %ymm0

    // Now, loop.
    xorq    %rcx, %rcx
.begin_vector_loop_1:
    cmpq    %rax, %rcx
    jge     .end_vector_loop_1

    // Load the data.
    vmovupd     (%rsi), %ymm1
    vmovupd     (%rdx), %ymm2
    vmovupd     (%rbx), %ymm3

    // Scale the data.
    vmulpd  %ymm0, %ymm1, %ymm1

    // Multiply and add.
    vfmadd231pd     %ymm1, %ymm2, %ymm3

    // Store the data.
    vmovupd     %ymm3, (%rbx)

    // Increment rcx.
    incq    %rcx
    leaq    32(%rsi), %rsi
    leaq    32(%rdx), %rdx
    leaq    32(%rbx), %rbx

    jmp     .begin_vector_loop_1
.end_vector_loop_1:

    // Now handle the rest of the elements.
    popq    %rax
    xorb    %cl, %cl

.begin_unvector_loop_1:
    cmpb    %al, %cl
    jge     .end_unvector_loop_1

    // Load values.
    movsd   (%rsi), %xmm1
    movsd   (%rdx), %xmm2
    movsd   (%rbx), %xmm3

    // Scale by the prefactors.
    mulsd   %xmm0, %xmm1

    // Multiply and add. Backwards because AT&T, so xmm1 * xmm2 + xmm3 = xmm3.
    vfmadd231sd %xmm1, %xmm2, %xmm3

    // Save
    movsd   %xmm3, (%rbx)

    // Increment.
    incb    %cl
    leaq    4(%rsi), %rsi
    leaq    4(%rdx), %rdx
    leaq    4(%rbx), %rbx
    jmp     .begin_unvector_loop_1
.end_unvector_loop_1:

    movq    -8(%rbp), %rdi
    movq    -16(%rbp), %rsi
    movq    -24(%rbp), %rdx
    movq    -32(%rbp), %rcx
    movq    -40(%rbp), %rax
    movq    -48(%rbp), %rbx
    movupd  -64(%rbp), %xmm0
    vmovupd -96(%rbp), %ymm1
    vmovupd -128(%rbp), %ymm2
    vmovupd -160(%rbp), %ymm3


    // Undo the stack frame.
    addq    $160, %rsp
    popq    %rbp
    retq
    .cfi_endproc
