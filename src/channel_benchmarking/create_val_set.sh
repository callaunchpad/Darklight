while IFS='' read -r line || [[ -n "$line" ]]; do
	#echo $line
	mails=$(echo $line | tr " ")
	for i in $mails
	do
		echo $i
	done
	#arr=($line)
	#$short = ${arr[0]}
	#$long = ${arr[1]}
	echo $short
	mv "$short" './dataset/Sony_val/short/'
	mv "$long" './dataset/Sony_val/long/'
done < "./dataset/Sony_val_list.txt"
