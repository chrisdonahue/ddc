for COLL in fraxtil itg
do
	echo "Executing ${1} for ${COLL}"
	${1} ${COLL}
	echo "--------------------------------------------"
done
