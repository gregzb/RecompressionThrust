#include <iostream>
#include <cuda_runtime.h>
#include <thrust/advance.h>

int main()
{
    // Get the number of CUDA-capable devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s):\n\n";

    // Loop through each device and print properties and stats
    for (int device = 0; device < deviceCount; device++)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess)
        {
            std::cerr << "Error getting properties for device " << device << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }    

        std::cout << "=== Device " << device << " : " << prop.name << " ===\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << "\n";
        std::cout << "Clock Rate: " << (prop.clockRate * 1e-3) << " MHz\n";
        std::cout << "Memory Clock Rate: " << (prop.memoryClockRate * 1e-3) << " MHz\n";
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "L2 Cache Size: " << (prop.l2CacheSize / 1024) << " KB\n";
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";

        // Set the current device to ensure we get the proper memory info
        err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            std::cerr << "Error setting device " << device << ": " << cudaGetErrorString(err) << "\n";
            continue;
        }

        // Get free and total memory on the device
        size_t freeMem = 0, totalMem = 0;
        err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess)
        {
            std::cerr << "Error getting memory info for device " << device << ": "
                      << cudaGetErrorString(err) << "\n";
        }
        else
        {
            std::cout << "Free Global Memory: " << (freeMem / (1024 * 1024)) << " MB\n";
            std::cout << "Total Global Memory: " << (totalMem / (1024 * 1024)) << " MB\n";
        }

        std::cout << std::endl;
    }

    // Optionally, output CUDA runtime and driver versions.
    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000.0 << "\n";
    std::cout << "CUDA Driver Version: " << driverVersion / 1000.0 << "\n";

    return 0;
}