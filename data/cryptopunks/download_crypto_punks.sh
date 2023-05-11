offset=0
limit=100
total_rows=10000  # Set the total number of rows you want to retrieve

while [ $offset -lt $total_rows ]; do
    curl -X GET "https://datasets-server.huggingface.co/rows?dataset=huggingnft%2Fcryptopunks&config=huggingnft--cryptopunks&split=train&offset=$offset&limit=$limit" -o "output_$offset.json"
    offset=$((offset + limit))
done