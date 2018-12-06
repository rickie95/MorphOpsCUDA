################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../CUDAUtils.cu \
../MorphableOperator.cu \
../main.cu 

CPP_SRCS += \
../Image.cpp \
../PPM.cpp \
../StructuringElement.cpp 

OBJS += \
./CUDAUtils.o \
./Image.o \
./MorphableOperator.o \
./PPM.o \
./StructuringElement.o \
./main.o 

CU_DEPS += \
./CUDAUtils.d \
./MorphableOperator.d \
./main.d 

CPP_DEPS += \
./Image.d \
./PPM.d \
./StructuringElement.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_37,code=sm_37  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_37,code=compute_37 -gencode arch=compute_37,code=sm_37  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_37,code=sm_37  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.0/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


