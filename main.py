from readXls import ReadXls

xls = ReadXls("data/AllDiscussionData.xls")
print(xls.get_column_with_name("Message"))
