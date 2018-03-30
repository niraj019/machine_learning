#!/usr/bin/env python


MAL_DB1 = open("data.csv", encoding="utf8")

"""SOURCE1: http://mirror1.malwaredomains.com/files/domains.txt
    This script extract url from this source and categorize information regarding the url
    This script is not open source and should not be used without consulting the owner first
    any commercial use of this script is illegal without a direct authorization of the current owner.    
"""

class UrlGatheringInfo:

    def extract_lexical_information(self, url):
        ''''
        This function return lexical information
            args:
                url = str
            return:
                 url_length, dots, digits, dasheses = int
        '''
        url_length = len(url)
        dasheses = 0
        dots = 0
        digits = 0
        slashes = 0
        for details in url:
            if details == ".":
                dots += 1
            elif details == "-":
                dasheses += 1
            elif details == "/":
                slashes += 1
            elif details.isdigit():
                digits += 1
        #print(dasheses)
        return url_length, dots, digits, dasheses

    def extract_url(self, DB):
        ''''
        This function open a csv file and xtract the url

            args:
                DB = cvs file
            return:
                None

        '''
        counter = 0
        with open("malicious_vectors_1000.csv", "w") as mv:
            for x in DB:
                if counter <= 1: #skip line(s), useful if you have dataset with long headers
                    pass
                #elif counter >=42767 and counter <= 311948:
                elif (counter >=1000 and counter <= 42767) or (counter >=43767):
                    pass
                else:
                    pass_two = x.split(",")
                    #print(pass_two)
                    url = pass_two[0]
                    type = pass_two[1]
                    lex_info = self.extract_lexical_information(url)
                    #line = "{},{},{}".format(lex_info[0], lex_info[1], type)
                    line = "{},{},{},{},{}".format(lex_info[0], lex_info[1], lex_info[2], lex_info[3], type)
                    #print(len(line))
                    if len(line) > 15:
                        continue
                    else:
                        mv.writelines(line)
                counter += 1
        mv.close()


def main():
    extractor = UrlGatheringInfo()
    extractor.extract_url(MAL_DB1)


if __name__ == "__main__":
    main()
