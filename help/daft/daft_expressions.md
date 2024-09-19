Expressions
Daft Expressions allow you to express some computation that needs to happen in a DataFrame.

This page provides an overview of all the functionality that is provided by Daft Expressions.

Constructors
col

Creates an Expression referring to the column with the provided name.

lit

Creates an Expression representing a column with every value set to the provided value

Generic
to_struct

Converts multiple input expressions or column names into a struct.

Expression.alias

Gives the expression a new name, which is its column's name in the DataFrame schema and the name by which subsequent expressions can refer to the results of this expression.

Expression.cast

Casts an expression to the given datatype if possible.

Expression.if_else

Conditionally choose values between two expressions using the current boolean expression as a condition

Expression.is_null

Checks if values in the Expression are Null (a special value indicating missing data)

Expression.not_null

Checks if values in the Expression are not Null (a special value indicating missing data)

Expression.fill_null

Fills null values in the Expression with the provided fill_value

Expression.hash

Hashes the values in the Expression.

Expression.apply

Apply a function on each value in a given expression

Numeric
Expression.__abs__

Absolute of a numeric expression (abs(expr))

Expression.__add__

Adds two numeric expressions or concatenates two string expressions (e1 + e2)

Expression.__sub__

Subtracts two numeric expressions (e1 - e2)

Expression.__mul__

Multiplies two numeric expressions (e1 * e2)

Expression.__truediv__

True divides two numeric expressions (e1 / e2)

Expression.__mod__

Takes the mod of two numeric expressions (e1 % e2)

Expression.__lshift__

Shifts the bits of an integer expression to the left (e1 << e2) :param other: The number of bits to shift the expression to the left

Expression.__rshift__

Shifts the bits of an integer expression to the right (e1 >> e2)

Expression.ceil

The ceiling of a numeric expression (expr.ceil())

Expression.floor

The floor of a numeric expression (expr.floor())

Expression.sign

The sign of a numeric expression (expr.sign())

Expression.round

The round of a numeric expression (expr.round(decimals = 0))

Expression.sqrt

The square root of a numeric expression (expr.sqrt())

Expression.cbrt

The cube root of a numeric expression (expr.cbrt())

Expression.sin

The elementwise sine of a numeric expression (expr.sin())

Expression.cos

The elementwise cosine of a numeric expression (expr.cos())

Expression.tan

The elementwise tangent of a numeric expression (expr.tan())

Expression.cot

The elementwise cotangent of a numeric expression (expr.cot())

Expression.arcsin

The elementwise arc sine of a numeric expression (expr.arcsin())

Expression.arccos

The elementwise arc cosine of a numeric expression (expr.arccos())

Expression.arctan

The elementwise arc tangent of a numeric expression (expr.arctan())

Expression.arctan2

Calculates the four quadrant arctangent of coordinates (y, x), in radians (expr_y.arctan2(expr_x))

Expression.arctanh

The elementwise inverse hyperbolic tangent of a numeric expression (expr.arctanh())

Expression.arccosh

The elementwise inverse hyperbolic cosine of a numeric expression (expr.arccosh())

Expression.arcsinh

The elementwise inverse hyperbolic sine of a numeric expression (expr.arcsinh())

Expression.radians

The elementwise radians of a numeric expression (expr.radians())

Expression.degrees

The elementwise degrees of a numeric expression (expr.degrees())

Expression.log2

The elementwise log base 2 of a numeric expression (expr.log2())

Expression.log10

The elementwise log base 10 of a numeric expression (expr.log10())

Expression.log

The elementwise log with given base, of a numeric expression (expr.log(base = math.e)) :param base: The base of the logarithm.

Expression.ln

The elementwise natural log of a numeric expression (expr.ln())

Expression.exp

The e^self of a numeric expression (expr.exp())

Expression.shift_left

Shifts the bits of an integer expression to the left (expr << other) :param other: The number of bits to shift the expression to the left

Expression.shift_right

Shifts the bits of an integer expression to the right (expr >> other)

Logical
Expression.__invert__

Inverts a boolean expression (~e)

Expression.__and__

Takes the logical AND of two boolean expressions, or bitwise AND of two integer expressions (e1 & e2)

Expression.__or__

Takes the logical OR of two boolean or integer expressions, or bitwise OR of two integer expressions (e1 | e2)

Expression.__lt__

Compares if an expression is less than another (e1 < e2)

Expression.__le__

Compares if an expression is less than or equal to another (e1 <= e2)

Expression.__eq__

Compares if an expression is equal to another (e1 == e2)

Expression.__ne__

Compares if an expression is not equal to another (e1 != e2)

Expression.__gt__

Compares if an expression is greater than another (e1 > e2)

Expression.__ge__

Compares if an expression is greater than or equal to another (e1 >= e2)

Expression.between

Checks if values in the Expression are between lower and upper, inclusive.

Expression.is_in

Checks if values in the Expression are in the provided list

Expression.minhash

Runs the MinHash algorithm on the series.

Aggregation
The following can be used with DataFrame.agg or GroupedDataFrame.agg

Expression.count([mode])

Counts the number of values in the expression.

Expression.sum()

Calculates the sum of the values in the expression

Expression.mean()

Calculates the mean of the values in the expression

Expression.min()

Calculates the minimum value in the expression

Expression.max()

Calculates the maximum value in the expression

Expression.any_value([ignore_nulls])

Returns any value in the expression

Expression.agg_list()

Aggregates the values in the expression into a list

Expression.agg_concat()

Aggregates the values in the expression into a single string by concatenating them

Expression.approx_percentiles(percentiles)

Calculates the approximate percentile(s) for a column of numeric values

Expression.approx_count_distinct()

Calculates the approximate number of non-NULL unique values in the expression.

Strings
The following methods are available under the expr.str attribute.

Expression.str.contains

Checks whether each string contains the given pattern in a string column

Expression.str.match

Checks whether each string matches the given regular expression pattern in a string column

Expression.str.startswith

Checks whether each string starts with the given pattern in a string column

Expression.str.endswith

Checks whether each string ends with the given pattern in a string column

Expression.str.concat

Concatenates two string expressions together

Expression.str.split

Splits each string on the given literal or regex pattern, into a list of strings.

Expression.str.extract

Extracts the specified match group from the first regex match in each string in a string column.

Expression.str.extract_all

Extracts the specified match group from all regex matches in each string in a string column.

Expression.str.replace

Replaces all occurrences of a pattern in a string column with a replacement string.

Expression.str.length

Retrieves the length for a UTF-8 string column

Expression.str.length_bytes

Retrieves the length for a UTF-8 string column in bytes.

Expression.str.lower

Convert UTF-8 string to all lowercase

Expression.str.upper

Convert UTF-8 string to all upper

Expression.str.lstrip

Strip whitespace from the left side of a UTF-8 string

Expression.str.rstrip

Strip whitespace from the right side of a UTF-8 string

Expression.str.reverse

Reverse a UTF-8 string

Expression.str.capitalize

Capitalize a UTF-8 string

Expression.str.left

Gets the n (from nchars) left-most characters of each string

Expression.str.right

Gets the n (from nchars) right-most characters of each string

Expression.str.find

Returns the index of the first occurrence of the substring in each string

Expression.str.rpad

Right-pads each string by truncating or padding with the character

Expression.str.lpad

Left-pads each string by truncating on the right or padding with the character

Expression.str.repeat

Repeats each string n times

Expression.str.like

Checks whether each string matches the given SQL LIKE pattern, case sensitive

Expression.str.ilike

Checks whether each string matches the given SQL LIKE pattern, case insensitive

Expression.str.substr

Extract a substring from a string, starting at a specified index and extending for a given length.

Expression.str.to_date

Converts a string to a date using the specified format

Expression.str.to_datetime

Converts a string to a datetime using the specified format and timezone

Expression.str.normalize

Normalizes a string for more useful deduplication.

Expression.str.tokenize_encode

Encodes each string as a list of integer tokens using a tokenizer.

Expression.str.tokenize_decode

Decodes each list of integer tokens into a string using a tokenizer.

Expression.str.count_matches

Counts the number of times a pattern, or multiple patterns, appear in a string.

Floats
The following methods are available under the expr.float attribute.

Expression.float.is_inf

Checks if values in the Expression are Infinity.

Expression.float.is_nan

Checks if values are NaN (a special float value indicating not-a-number)

Expression.float.not_nan

Checks if values are not NaN (a special float value indicating not-a-number)

Expression.float.fill_nan

Fills NaN values in the Expression with the provided fill_value

Temporal
Expression.dt.date

Retrieves the date for a datetime column

Expression.dt.hour

Retrieves the day for a datetime column

Expression.dt.minute

Retrieves the minute for a datetime column

Expression.dt.second

Retrieves the second for a datetime column

Expression.dt.time

Retrieves the time for a datetime column

Expression.dt.day

Retrieves the day for a datetime column

Expression.dt.month

Retrieves the month for a datetime column

Expression.dt.year

Retrieves the year for a datetime column

Expression.dt.day_of_week

Retrieves the day of the week for a datetime column, starting at 0 for Monday and ending at 6 for Sunday

Expression.dt.truncate

Truncates the datetime column to the specified interval

List
Expression.list.join

Joins every element of a list using the specified string delimiter

Expression.list.lengths

Gets the length of each list

Expression.list.get

Gets the element at an index in each list

Expression.list.slice

Gets a subset of each list

Expression.list.chunk

Splits each list into chunks of the given size

Expression.list.sort

Sorts the inner lists of a list column.

Struct
Expression.struct.get

Retrieves one field from a struct column

Map
Expression.map.get

Retrieves the value for a key in a map column

Image
Expression.image.decode

Decodes the binary data in this column into images.

Expression.image.encode

Encode an image column as the provided image file format, returning a binary column of encoded bytes.

Expression.image.resize

Resize image into the provided width and height.

Expression.image.crop

Crops images with the provided bounding box

Expression.image.to_mode

Partitioning
Expression.partitioning.days

Partitioning Transform that returns the number of days since epoch (1970-01-01)

Expression.partitioning.hours

Partitioning Transform that returns the number of hours since epoch (1970-01-01)

Expression.partitioning.months

Partitioning Transform that returns the number of months since epoch (1970-01-01)

Expression.partitioning.years

Partitioning Transform that returns the number of years since epoch (1970-01-01)

Expression.partitioning.iceberg_bucket

Partitioning Transform that returns the Hash Bucket following the Iceberg Specification of murmur3_32_x86 https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements

Expression.partitioning.iceberg_truncate

Partitioning Transform that truncates the input to a standard width w following the Iceberg Specification.

URLs
Expression.url.download

Treats each string as a URL, and downloads the bytes contents as a bytes column

JSON
Expression.json.query

Query JSON data in a column using a JQ-style filter https://jqlang.github.io/jq/manual/ This expression uses jaq as the underlying executor, see 01mf02/jaq for the full list of supported filters.

Embedding
Expression.embedding.cosine_distance

Compute the cosine distance between two embeddings