################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../CUDAUtils.cu \
../main.cu 

OBJS += \
./CUDAUtils.o \
./main.o 

CU_DEPS += \
./CUDAUtils.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 -gencode arch=compute_37,code=sm_37  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_37,code=compute_37 -gencode arch=compute_37,code=sm_37  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


