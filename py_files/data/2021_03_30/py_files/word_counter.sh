C=0
for i in $(find . -name "*.py")
do
	A=`wc -l $i | cut -f1 -d' '`
	C=`expr $C + $A`
	echo $i
	echo $A
done
echo "Total:"
echo $C
