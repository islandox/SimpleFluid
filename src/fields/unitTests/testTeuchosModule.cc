import Trilinos.Teuchos;

using rcp_type = Teuchos::RCP<int>;

static_assert(sizeof(rcp_type) > 0);

int main()
{
    return 0;
}
