
from nmem.analysis.core_analysis import calculate_operating_table
from nmem.analysis.data_import import (
    import_operating_data,
)


def main():
    dict_list, ic_list, write_current_list, ic_list2, write_current_list2 = (
        import_operating_data("write_current_sweep_C3")
    )
    df = calculate_operating_table(
        dict_list, ic_list, write_current_list, ic_list2, write_current_list2
    )
    save_csv = False
    if save_csv:
        df.to_csv("read_current_sweep_operating.csv", float_format="%.3f")
    else:
        print(df)


if __name__ == "__main__":
    main()
