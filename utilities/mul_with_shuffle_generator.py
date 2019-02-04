def gen_tables(array_len):
    even_table = {}
    odd_table = {}
    even_table_size = 0
    odd_table_size = 0
    
    #initialize dictionaries
    for i in xrange(array_len):
        even_table[i] = []
        odd_table[i] = []
    
    for a in xrange(array_len):
        for b in xrange(array_len):
            idx = (a + b) / 2
            if (a + b) % 2 == 0:
                even_table[idx].append((a, b))
                even_table_size = even_table_size + 1
            else:
                odd_table[idx].append((a, b))
                odd_table_size = odd_table_size + 1
                
    return (even_table, odd_table, even_table_size, odd_table_size)

from collections import namedtuple

#c = a*b + d
AsmInsn = namedtuple("AsmInsn", "is_even op_type gen_carry use_carry dest_op first_op second_op")
op_mul = 0
op_add = 1
op_shfl = 2
op_unknown = 3

def find_largest_sublist(table, array_len):
    count = -1
    indexes = []
    for j in xrange(array_len):
        if len(table[j]) > count:
            count = len(table[j])
            indexes = []
        else if len(table[j]) == count:
            indexes.append[j]
    return count, indexes

def gen_asm(array_len):
    AsmListing = []
    
    even_table, odd_table, even_table_size, odd_table_size = gen_tables(array_len)
    
    #do the same procedure - first for even table, than to odd
    
    for (table in (even_table, odd_table)):
        is_even = (table == even_table)
                
        #fill every register

        for i in xrange(array_len):
            (a, b) = even_table[i][0]
            even_table[i].pop(0)
            insn = AsmListing(is_even, op_mul, False, False, i, a, b)
            even_table_size = even_table_size - 1

        while(even_table_size > 0):
            count, indexes = find_largest_sublist(even_table_size, array_len)
            iter_pos = 0
            iter_end = len(indexes)

            use_carry = False

            while (iter_pos < iter_end):

            cur_index = indexes[0]

            (a, b) = even_table[cur_index][0]
            even_table[cur_index].pop(0)
            insn = AsmListing(op_add, False, False, cur_index, a, b)
            even_table_size = even_table_size - 1

            if (count > 1)
        
    #now generate code for final shifting and adding
    
    "shf.l.clamp.b32 s7, t6, t7, 16;\n\t"
         "shf.l.clamp.b32 s6, t5, t6, 16;\n\t"
         "shf.l.clamp.b32 s5, t4, t5, 16;\n\t"
         "shf.l.clamp.b32 s4, t3, t4, 16;\n\t"
         "shf.l.clamp.b32 s3, t2, t3, 16;\n\t"
         "shf.l.clamp.b32 s2, t1, t2, 16;\n\t"
         "shf.l.clamp.b32 s1, t0, t1, 16;\n\t"
         "shf.l.clamp.b32 s0,  0, t0, 16;\n\t"
 
   

def generate_printable_asm(AsmListing, res_reg_name = "r", temp_reg_name1 = "t", temp_reg_name2 = "s"):
    printed_asm = ""
    ending = "\\n\\t"

    for elem in AsmListing:
        if (elem.op_type == op_mul):         
            printed_asm += "\"mul." + high_low + "u32";
            printed_asm += '   %{:d}, %{:d}, %{:d};\\n\\t\"\n'.format(elem.c, elem.a, elem.b)

        elif (elem.op_type == op_add):
            printed_asm += "\"add"
            if (elem.use_carry):
                printed_asm += "c"
            printed_asm += "."
            if (elem.gen_carry):
                printed_asm += "cc."
            printed_asm += "u32" + '   %{:d}, %{:d}, 0;\\n\\t\"\n'.format(elem.c, elem.a)
        
        elif (elem.op_type == op_shfl):
        
        else:
            raise ValueError('Incorrect operand type.')
     
    return printed_asm