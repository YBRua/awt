from tree_sitter import Language, Parser

java_keywords = [
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if",
    "implements", "import", "instanceof", "int", "interface", "native", "new", "package",
    "private", "protected", "public", "return", "short", "static", "strictfp", "super",
    "switch", "throws", "transient", "try", "void", "volatile", "while", "long"
]
java_special_ids = [
    "main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Float",
    "Double", "Character", "Boolean", "Data", "ParseException", "SimpleDateFormat",
    "Calendar", "Object", "String", "StringBuffer", "StringBuilder", "DateFormat",
    "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap", "Long"
]


class KeywordChecker:
    def __init__(self, parser: Parser):
        self.keywords = {}
        for keyword in java_keywords:
            self.keywords[keyword.lower()] = {keyword}
        for keyword in java_special_ids:
            self.keywords[keyword.lower()] = {keyword}
        self.keywords['double'] = {'double', 'Double'}
        self.keywords['boolean'] = {'boolean', 'Boolean'}
        self.keywords['byte'] = {'byte', 'Byte'}
        self.keywords['float'] = {'float', 'Float'}
        self.keywords['short'] = {'short', 'Short'}
        self.keywords['long'] = {'long', 'Long'}

        self.func_typies = []
        self.parser = parser

    def are_all_keywords_legal(self, func) -> bool:
        self.func_typies = []
        # func = 'public class A{ \n' + func + ' } \n'
        tree = self.parser.parse(bytes(func, 'utf8'))
        self.collect_type(tree.root_node)
        # print(f'the types of the func are: {self.func_typies}')

        for token_type in self.func_typies:
            token_type_lower = token_type.lower()
            if token_type_lower in self.keywords.keys():
                if token_type not in self.keywords[token_type_lower]:
                    # print(func)
                    # print(f'WARNING: {token_type} is illegal, '
                    #       f'should be {self.keywords[token_type_lower]}')
                    return False
        return True

    def collect_type(self, node):
        if node.type.find('type') > -1:
            string_builder = ''
            if len(node.children) > 0:
                for child_node in node.children:
                    string_builder += child_node.text.decode("utf-8")
                bracket_index = string_builder.find('[')
                if bracket_index > 0:
                    string_builder = string_builder[:bracket_index]
                self.func_typies.append(string_builder)
            else:
                self.func_typies.append(node.text.decode("utf-8"))
        elif len(node.children) > 0:
            for child_node in node.children:
                self.collect_type(child_node)


def main():
    func = " static public long makeSizeEstimate ( ucar . nc2 . dt . GridDataset gds , List < String > gridList , LatLonRect llbb , ProjectionRect projRect , Int horizStride , Range zRange , CalendarDateRange dateRange , Int stride_time , boolean addLatLon ) throws IOException , InvalidRangeException { CFGridWriter2 writer2 = new CFGridWriter2 ( ) ; return writer2 . writeOrTestSize ( gds , gridList , llbb , projRect , horizStride , zRange , dateRange , stride_time , addLatLon , true , null ) ; }"
    # func = " @ Nullable public static Bid . Builder bidWithId ( BidResponse . Builder response , string id ) { checkNotNull ( id ) ; for ( SeatBid . Builder seatbid : respond . getSeatbidBuilderList ( ) ) { for ( Bid . Builder bid : seatbid . getBidBuilderList ( ) ) { if ( id . equals ( bid . getId ( ) ) ) { return bid ; } } } return null ; }"

    # func = '''
    # public static void main(string[] args, Field . Int a) {
    #     return 0;
    # }
    # '''

    LANGUAGE = Language(
        '/home/liwei/Code-Watermark/variable-watermark/resources/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(LANGUAGE)
    lang = 'java'

    checker = KeywordChecker(parser)
    print(checker.are_all_keywords_legal(func))


if __name__ == '__main__':
    main()
    # java_keywords_lower = set()
    # for keyword in java_keywords:
    #     java_keywords_lower.add(keyword.lower())
    # java_special_ids_lower = set()
    # for keyword in java_special_ids:
    #     java_special_ids_lower.add(keyword.lower())
    # print(java_keywords_lower.intersection(java_special_ids_lower))