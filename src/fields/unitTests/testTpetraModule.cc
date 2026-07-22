import SimpleFluid.Tpetra;

using map_type = Tpetra::Map<int, long long>;

static_assert(sizeof(map_type) > 0);

int main()
{
    return 0;
}
