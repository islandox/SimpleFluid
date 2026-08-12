import Trilinos.Zoltan2;

using graph_type = Tpetra::CrsGraph<int, long long>;
using adapter_type = Zoltan2::TpetraRowGraphAdapter<graph_type>;

static_assert(sizeof(adapter_type) > 0);

int main()
{
    return 0;
}
