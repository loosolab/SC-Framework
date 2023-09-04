"""Custom Datatypes for type hinting"""
import pandera as pa

class _pandas_dataframe(pa.DataFrameModel):
    """Util class for dataframe typecheck"""
    pass