// Keep this header-unit consumer free of textual standard-library and GTest
// headers; header units do not propagate their include-guard macros.
import SimpleFluid.Kokkos;
import SimpleFluid.Teuchos;
import SimpleFluid.Tpetra;
import SimpleFluid.Zoltan2;

using view_type = Kokkos::View<double*>;
using rcp_type = Teuchos::RCP<int>;
using map_type = Tpetra::Map<int, long long>;
using graph_type = Tpetra::CrsGraph<int, long long>;
using adapter_type = Zoltan2::TpetraRowGraphAdapter<graph_type>;

static_assert(std::is_class_v<view_type>);
static_assert(std::is_class_v<rcp_type>);
static_assert(std::is_class_v<map_type>);
static_assert(std::is_class_v<adapter_type>);

int main()
{
    return 0;
}
