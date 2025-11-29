#include "plast/tensor/tensor.h"
#include "plast/core/types.h"
#include <iostream>

#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace plast
{
namespace tensor
{

// Helper to get size of DType in bytes
size_t get_dtype_size(core::DType dtype)
{
    switch (dtype)
    {
    case core::DType::FLOAT32:
        return sizeof(float);
    case core::DType::FLOAT64:
        return sizeof(double);
    case core::DType::INT8:
        return sizeof(int8_t);
    case core::DType::INT16:
        return sizeof(int16_t);
    case core::DType::INT32:
        return sizeof(int32_t);
    case core::DType::INT64:
        return sizeof(int64_t);
    case core::DType::UINT8:
        return sizeof(uint8_t);
    case core::DType::UINT16:
        return sizeof(uint16_t);
    case core::DType::UINT32:
        return sizeof(uint32_t);
    case core::DType::UINT64:
        return sizeof(uint64_t);
    case core::DType::BOOL:
        return sizeof(bool);
    case core::DType::UNKNOWN:
    default:
        throw std::runtime_error("Unknown or unsupported DType.");
    }
}

// Helper to calculate contiguous strides
std::vector<size_t> calculate_contiguous_strides(const std::vector<size_t>& shape)
{
    std::vector<size_t> strides(shape.size());
    if (shape.empty())
    {
        return strides;
    }
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Constructor with existing data
Tensor::Tensor(void* data, const std::vector<size_t>& shape, const std::vector<size_t>& strides,
               core::DType dtype, core::DeviceType device, bool owns_data)
    : data_(data), shape_(shape), strides_(strides), dtype_(dtype), device_(device),
      owns_data_(owns_data)
{
    if (shape_.empty())
    {
        shape_.push_back(1); // Scalar tensor
        strides_.push_back(1);
    }
    if (shape_.size() != strides_.size())
    {
        throw std::runtime_error("Shape and strides must have the same number of dimensions.");
    }
}

// Constructor for empty tensor (allocates memory)
Tensor::Tensor(const std::vector<size_t>& shape, core::DType dtype, core::DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device), owns_data_(true)
{
    if (shape_.empty())
    {
        shape_.push_back(1); // Scalar tensor
    }
    strides_ = calculate_contiguous_strides(shape_);
    allocate_data();
}

// Destructor
Tensor::~Tensor() { deallocate_data(); }

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      dtype_(other.dtype_), device_(other.device_), owns_data_(other.owns_data_)
{
    other.data_ = nullptr;
    other.owns_data_ = false;
}

// Move assignment operator
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        deallocate_data(); // Deallocate current resources

        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;

        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

size_t Tensor::num_elements() const
{
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

size_t Tensor::nbytes() const { return num_elements() * get_dtype_size(dtype_); }

bool Tensor::is_contiguous() const
{
    if (shape_.empty())
    {
        return true; // Scalar is contiguous
    }

    size_t expected_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (strides_[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

void Tensor::allocate_data()
{
    size_t bytes = nbytes();
    if (bytes == 0)
    {
        data_ = nullptr;
        return;
    }

    switch (device_)
    {
    case core::DeviceType::CPU:
        data_ = new char[bytes];
        break;
    case core::DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
        if (cudaMalloc(&data_, bytes) != cudaSuccess)
        {
            throw std::runtime_error("CUDA memory allocation failed.");
        }
#else
        throw std::runtime_error("CUDA is not enabled. Cannot allocate CUDA memory.");
#endif
        break;
    default:
        throw std::runtime_error("Unsupported device type for allocation.");
    }
}

void Tensor::deallocate_data()
{
    if (data_ && owns_data_)
    {
        switch (device_)
        {
        case core::DeviceType::CPU:
            delete[] static_cast<char*>(data_);
            break;
        case core::DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
            if (cudaFree(data_) != cudaSuccess)
            {
                // Log error but don't throw in destructor
                std::cerr << "CUDA memory deallocation failed." << std::endl;
            }
#else
            // Should not happen if allocation checked PLAST_CUDA_ENABLED
#endif
            break;
        default:
            // Should not happen
            break;
        }
        data_ = nullptr;
    }
}

void Tensor::copy_data_from(const Tensor& other)
{
    if (this == &other) return;

    if (nbytes() != other.nbytes())
    {
        throw std::runtime_error("Cannot copy data: byte sizes differ.");
    }

    if (data_ == nullptr)
    { // If this tensor doesn't own data or hasn't allocated yet
        allocate_data();
    }

    if (device_ == other.device_)
    {
        // Same device copy
        switch (device_)
        {
        case core::DeviceType::CPU:
            std::memcpy(data_, other.data_, nbytes());
            break;
        case core::DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
            if (cudaMemcpy(data_, other.data_, nbytes(), cudaMemcpyDeviceToDevice) != cudaSuccess)
            {
                throw std::runtime_error("CUDA device to device memcpy failed.");
            }
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported device type for same-device copy.");
        }
    }
    else
    {
        // Cross-device copy
        if (device_ == core::DeviceType::CPU && other.device_ == core::DeviceType::CUDA)
        {
#ifdef PLAST_CUDA_ENABLED
            if (cudaMemcpy(data_, other.data_, nbytes(), cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                throw std::runtime_error("CUDA device to host memcpy failed.");
            }
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
        }
        else if (device_ == core::DeviceType::CUDA && other.device_ == core::DeviceType::CPU)
        {
#ifdef PLAST_CUDA_ENABLED
            if (cudaMemcpy(data_, other.data_, nbytes(), cudaMemcpyHostToDevice) != cudaSuccess)
            {
                throw std::runtime_error("CUDA host to device memcpy failed.");
            }
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
        }
        else
        {
            throw std::runtime_error("Unsupported cross-device copy combination.");
        }
    }
}

Tensor Tensor::to(core::DeviceType target_device) const
{
    if (device_ == target_device)
    {
        return clone(); // Already on target device, return a copy
    }

    Tensor new_tensor(shape_, dtype_, target_device);
    new_tensor.copy_data_from(*this);
    return new_tensor;
}

Tensor Tensor::clone() const
{
    // Create a new contiguous tensor with the same shape, dtype, and device
    Tensor new_tensor(shape_, dtype_, device_);

    if (is_contiguous() && new_tensor.is_contiguous())
    {
        // If both are contiguous, a simple data copy is sufficient
        new_tensor.copy_data_from(*this);
    }
    else
    {
        // If not contiguous, iterate through elements and copy them
        size_t num_elements = this->num_elements();
        size_t item_size = get_dtype_size(dtype_);

        // Get raw pointers to data
        char* src_data = static_cast<char*>(data_);
        char* dst_data = static_cast<char*>(new_tensor.data_);

        std::vector<size_t> current_coords(shape_.size(), 0);
        for (size_t i = 0; i < num_elements; ++i)
        {
            size_t src_offset = 0;
            for (size_t dim = 0; dim < shape_.size(); ++dim)
            {
                src_offset += current_coords[dim] * strides_[dim];
            }

            // Copy element from source to destination
            std::memcpy(dst_data + i * item_size, src_data + src_offset * item_size, item_size);

            // Increment coordinates for the next element
            for (int dim = shape_.size() - 1; dim >= 0; --dim)
            {
                current_coords[dim]++;
                if (current_coords[dim] < shape_[dim])
                {
                    break;
                }
                current_coords[dim] = 0;
            }
        }
    }
    return new_tensor;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const
{
    size_t new_num_elements =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_num_elements != num_elements())
    {
        throw std::runtime_error("Reshape operation requires the total number of elements to "
                                 "remain constant.");
    }

    // Calculate new contiguous strides for the new shape
    std::vector<size_t> new_strides = calculate_contiguous_strides(new_shape);

    // Create a new Tensor that views the same data, but with a new shape and new strides.
    // owns_data is set to false because the new tensor does not own the data.
    return Tensor(data_, new_shape, new_strides, dtype_, device_, false);
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape,
                       const std::vector<size_t>& new_strides) const
{
    size_t new_num_elements =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_num_elements != num_elements())
    {
        throw std::runtime_error("Reshape operation requires the total number of elements to "
                                 "remain constant.");
    }
    if (new_shape.size() != new_strides.size())
    {
        throw std::runtime_error("New shape and new strides must have the same number of "
                                 "dimensions for reshape.");
    }

    // Create a new Tensor that views the same data, but with a new shape and new strides.
    // owns_data is set to false because the new tensor does not own the data.
    return Tensor(data_, new_shape, new_strides, dtype_, device_, false);
}

} // namespace tensor
} // namespace plast
