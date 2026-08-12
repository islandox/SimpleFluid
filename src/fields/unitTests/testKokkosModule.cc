import Trilinos.Kokkos;

using view_type = Kokkos::View<double*>;

static_assert(sizeof(view_type) > 0);

int main()
{
    return 0;
}
