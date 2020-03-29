import xlrd


class ReadXls:

    def __init__(self, path):
        self.wb = xlrd.open_workbook(path)
        self.sheet = self.wb.sheet_by_index(0)
        self.sheet.cell_value(0, 0)

    def get_rows_number(self):
        return self.sheet.nrows

    def get_columns_number(self):
        return self.sheet.ncols

    def get_columns_name(self):
        return [self.sheet.cell_value(0, i) for i in range(self.sheet.ncols)]

    # without column name
    def get_column_at_index(self, index):
        # return [self.sheet.cell_value(i + 1, index) for i in range(self.sheet.nrows - 1)]
        sez = []
        for i in range(self.sheet.nrows - 1):
            if self.sheet.cell(i + 1, index).ctype == xlrd.XL_CELL_EMPTY:  # stops on empty cell
                continue
            sez.append(self.sheet.cell_value(i + 1, index))
        return sez

    def get_row_at_index(self, index):
        return self.sheet.row_values(index)

    def get_column_with_name(self, name):
        return self.get_column_at_index(self.get_columns_name().index(name))
