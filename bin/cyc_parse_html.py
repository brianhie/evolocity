from html.parser import HTMLParser
import sys

class OrthoTreeParser(HTMLParser):
    def __init__(self):
        super().__init__()

        self.tree_depth = 1
        self.curr_path = [ 'Eukaryota' ]
        self.look_for_node = False
        self.look_for_leaf = False
        self.look_for_data = False
        self.in_link = False

    def handle_starttag(self, tag, attrs):
        if tag == 'ul':
            self.tree_depth += 1

        elif tag == 'span':
            self.look_for_node = bool(sum([
                name == 'class' and val == 'tree-name tree-folder'
                for name, val in attrs
            ]))
            self.look_for_leaf = bool(sum([
                name == 'class' and val == 'tree-name'
                for name, val in attrs
            ]))

        elif tag == 'a':
            self.in_link = True

    def handle_data(self, data):
        data = data.strip()

        if self.look_for_node:
            self.curr_path.append(data)
            assert(self.tree_depth == len(self.curr_path))
            self.look_for_node = False

        if self.look_for_leaf:
            sys.stdout.write(data)
            self.look_for_leaf = False
            self.look_for_data = True

        if self.look_for_data and self.in_link:
            sys.stdout.write('\t' + data)

    def handle_endtag(self, tag):
        if tag == 'ul':
            self.tree_depth -= 1
            self.curr_path.pop()

        elif tag == 'li':
            if self.look_for_data:
                sys.stdout.write('\t' + ','.join(self.curr_path) + '\n')
            self.look_for_data = False

        elif tag == 'a':
            self.in_link = False


if __name__ == '__main__':
    parser = OrthoTreeParser()

    with open('data/cyc/cytochrome_c-like_domain_1533604.html') as f:
        html = f.read().strip()

    parser.feed(html)
