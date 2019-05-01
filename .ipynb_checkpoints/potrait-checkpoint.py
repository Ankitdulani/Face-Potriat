import xlrd

path=("face.xslx")
file =open(path,'r')

for line in file:
	if line.strip():           # line contains eol character(s)
		print(int(line)) 


# wb =xlrd.open_workbook(path)
# sheet=wb.sheet_by_index(0)

# print (sheet.neows + " " + sheet.nclos)

