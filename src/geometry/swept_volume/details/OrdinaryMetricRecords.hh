#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace SimpleFluid::geometry::swept_volume::details
{
/** Numerical storage only: neither a construction witness nor replay provenance.
 * Local point numbering, ring order, and all four weights/errors survive freezing.
 */
template<class Work, class Coefficient> class OrdinaryMetricRecords
{
public:
    using Weights = std::array<long double, 4>;
    struct Row
    {
        Weights weights, errors;
    };
    struct Face
    {
        std::size_t first, count;
        int source;
    };
    struct Channel
    {
        std::size_t local_first = 0, face_first = 0, face_count = 0;
    };
    struct Record
    {
        std::size_t tetra = 0;
        std::vector<Row> rows;
        std::vector<std::size_t> local_to_numeric;
        std::vector<Face> faces;
        std::vector<std::size_t> rings;
        std::array<Channel, 3> channels;
    };
    struct FaceView
    {
        std::span<const std::size_t> vertices;
        int source;
    };
    struct Faces
    {
        std::span<const Face> descriptors;
        std::span<const std::size_t> rings;
        struct Iterator
        {
            const Face* face;
            std::span<const std::size_t> rings;
            FaceView operator*() const { return {rings.subspan(face->first, face->count), face->source}; }
            Iterator& operator++()
            {
                ++face;
                return *this;
            }
            bool operator!=(const Iterator& other) const { return face != other.face; }
        };
        Iterator begin() const { return {descriptors.data(), rings}; }
        Iterator end() const
        {
            return {descriptors.empty() ? descriptors.data() : descriptors.data() + descriptors.size(), rings};
        }
        bool empty() const { return descriptors.empty(); }
        FaceView front() const { return *begin(); }
    };
    struct MetricView
    {
        const Record& record;
        std::size_t local_first;
        Faces faces;
        const Weights& weights(std::size_t local) const
        {
            return record.rows[record.local_to_numeric[local_first + local]].weights;
        }
        const Weights& errors(std::size_t local) const
        {
            return record.rows[record.local_to_numeric[local_first + local]].errors;
        }
    };

    explicit OrdinaryMetricRecords(Work& work) : work_(work)
    {
        work_.charge(sizeof(*this));
        observe();
    }
    OrdinaryMetricRecords(const OrdinaryMetricRecords&) = delete;
    ~OrdinaryMetricRecords()
    {
        work_.live_compact_metric_bytes -= observed_;
        work_.observe_geometry();
    }
    void reserve_records(std::size_t count, bool coefficients)
    {
        reserve(records_, count);
        if (coefficients)
            reserve(coefficients_, count);
    }
    /** All channels of this tetrahedron are live together. Sharing is issued
     * arena-slot identity within this freeze only, never coordinate equality. */
    template<class Poly>
    void freeze(
        std::size_t tetra, const std::array<const Poly*, 3>& polys, const std::array<Coefficient, 4>* coefficients)
    {
        std::size_t local_count = 0, face_count = 0, ring_count = 0;
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            const auto* p = polys[channel];
            if (!p || alias(polys, channel) != channel)
                continue;
            local_count = add(local_count, p->vertices.size());
            face_count = add(face_count, p->faces.size());
            for (const auto& face : p->faces)
                ring_count = add(ring_count, face.vertices.size());
        }
        // Check the complete union's byte arithmetic before any allocation.
        (void) multiply(local_count, sizeof(Row));
        (void) multiply(local_count, sizeof(std::size_t));
        (void) multiply(face_count, sizeof(Face));
        (void) multiply(ring_count, sizeof(std::size_t));
        Scratch scratch(work_);
        std::vector<std::size_t> slots;
        scratch.reserve(slots, local_count);
        for (unsigned channel = 0; channel < 3; ++channel)
            if (polys[channel] && alias(polys, channel) == channel)
                for (const auto& vertex : polys[channel]->vertices)
                    slots.push_back(vertex.slot_id().value);
        std::sort(slots.begin(), slots.end());
        slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
        if (records_.size() == records_.capacity())
            reserve(records_, add(records_.size(), 1));
        if (coefficients && coefficients_.size() == coefficients_.capacity())
            reserve(coefficients_, add(coefficients_.size(), 1));
        records_.emplace_back();
        auto& record = records_.back();
        record.tetra = tetra;
        reserve(record.rows, slots.size());
        reserve(record.local_to_numeric, local_count);
        reserve(record.faces, face_count);
        reserve(record.rings, ring_count);
        record.rows.resize(slots.size());
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            const auto* p = polys[channel];
            if (!p)
                continue;
            const auto original = alias(polys, channel);
            if (original != channel)
            {
                record.channels[channel] = record.channels[original];
                continue;
            }
            record.channels[channel] = {record.local_to_numeric.size(), record.faces.size(), p->faces.size()};
            for (std::size_t local = 0; local < p->vertices.size(); ++local)
            {
                const auto slot = p->vertices[local].slot_id().value;
                const auto numeric =
                    static_cast<std::size_t>(std::lower_bound(slots.begin(), slots.end(), slot) - slots.begin());
                record.local_to_numeric.push_back(numeric);
                record.rows[numeric] = {p->weights(local), p->errors(local)};
            }
            for (const auto& face : p->faces)
            {
                record.faces.push_back({record.rings.size(), face.vertices.size(), face.source});
                record.rings.insert(record.rings.end(), face.vertices.begin(), face.vertices.end());
            }
        }
        if (coefficients)
            coefficients_.push_back(*coefficients);
    }
    const std::vector<Record>& records() const { return records_; }
    const std::array<Coefficient, 4>& coefficients(std::size_t index) const { return coefficients_[index]; }
    std::size_t coefficient_count() const { return coefficients_.size(); }
    static MetricView view(const Record& record, unsigned channel)
    {
        const auto& c = record.channels[channel];
        return {record, c.local_first,
            {std::span<const Face>(record.faces).subspan(c.face_first, c.face_count), record.rings}};
    }

private:
    Work& work_;
    std::vector<Record> records_;
    std::vector<std::array<Coefficient, 4>> coefficients_;
    std::size_t observed_ = 0;
    [[noreturn]] static void overflow()
    {
        throw ::SimpleFluid::PLIC::GeometryError(::SimpleFluid::PLIC::GeometryErrorCode::ResourceLimit,
            "Ordinary compact metric storage size exceeds its represented range.");
    }
    static std::size_t add(std::size_t a, std::size_t b)
    {
        if (b > std::numeric_limits<std::size_t>::max() - a)
            overflow();
        return a + b;
    }
    static std::size_t multiply(std::size_t a, std::size_t b)
    {
        if (a && b > std::numeric_limits<std::size_t>::max() / a)
            overflow();
        return a * b;
    }
    template<class Poly> static unsigned alias(const std::array<const Poly*, 3>& polys, unsigned channel)
    {
        for (unsigned first = 0; first < channel; ++first)
            if (polys[first] == polys[channel])
                return first;
        return channel;
    }
    /** The overlap observer owns the replacement allocation until observations
     * transfer to its destination, and releases even after partial admissions. */
    struct Scratch
    {
        Work& work;
        std::size_t bytes = 0;
        bool metric = false;
        explicit Scratch(Work& w, bool m = false) : work(w), metric(m) {}
        ~Scratch() { set(0); }
        void set(std::size_t n) noexcept
        {
            auto& live = metric ? work.live_compact_metric_bytes : work.live_freeze_scratch_bytes;
            live = live - bytes + n;
            bytes = n;
            work.observe_geometry();
        }
        template<class T> void reserve(std::vector<T>& values, std::size_t count)
        {
            const auto requested = multiply(count, sizeof(T));
            work.charge(requested);
            set(requested);
            values.reserve(count);
            const auto actual = multiply(values.capacity(), sizeof(T));
            set(actual);
            if (actual != requested)
                work.charge(actual - requested);
        }
    };
    template<class T> void reserve(std::vector<T>& values, std::size_t count)
    {
        if (count <= values.capacity())
            return;
        Scratch replacement(work_, true);
        try
        {
            replacement.reserve(values, count);
        }
        catch (...)
        {
            replacement.set(0);
            observe();
            throw;
        }
        replacement.set(0);
        observe();
    }
    void observe() noexcept
    {
        auto bytes = sizeof(*this) + records_.capacity() * sizeof(Record) +
                     coefficients_.capacity() * sizeof(std::array<Coefficient, 4>);
        for (const auto& record : records_)
            bytes += record.rows.capacity() * sizeof(Row) + record.local_to_numeric.capacity() * sizeof(std::size_t) +
                     record.faces.capacity() * sizeof(Face) + record.rings.capacity() * sizeof(std::size_t);
        work_.live_compact_metric_bytes = work_.live_compact_metric_bytes - observed_ + bytes;
        observed_ = bytes;
        work_.observe_geometry();
    }
};
} // namespace SimpleFluid::geometry::swept_volume::details
