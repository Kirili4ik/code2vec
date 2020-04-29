import subprocess


class Extractor:
    def __init__(self, config, jar_path, max_path_length, max_path_width):
        self.config = config
        self.max_path_length = max_path_length
        self.max_path_width = max_path_width
        self.jar_path = jar_path

    def extract_paths(self, path):
        #command = ['java', '-jar', self.jar_path, 'code2vec', '--lang', 'py', '--project', 'pred_files', '--output', 'cd2vec' , '--maxH', str(self.max_path_length), '--maxW', str(self.max_path_width)]

        output_file = open('cd2vec/path_contexts_0.csv', 'r')
        
        #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out, err = process.communicate()
        #output = out.decode().splitlines()
        #if len(output_file) == 0:
        #    err = err.decode()
        #    raise ValueError(err)

        hash_to_string_dict = {}
        result = []
        for i, line in enumerate(output_file):
            parts = line.rstrip().split(' ')
            method_name = parts[0]
            current_result_line_parts = [method_name]
            contexts = parts[1:]
            for context in contexts[:self.config.MAX_CONTEXTS]:
                context_parts = context.split(',')
                context_word1 = context_parts[0]
                context_path = context_parts[1]
                context_word2 = context_parts[2]
                hashed_path = str(self.java_string_hashcode(context_path))
                hash_to_string_dict[hashed_path] = context_path
                current_result_line_parts += ['%s,%s,%s' % (context_word1, hashed_path, context_word2)]
            space_padding = ' ' * (self.config.MAX_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result, hash_to_string_dict

    @staticmethod
    def java_string_hashcode(s):
        """
        Imitating Java's String#hashCode, because the model is trained on hashed paths but we wish to
        Present the path attention on un-hashed paths.
        """
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
