import sys
from retrieval import retrieve
from generate import generate_answer, verify_citations


def answer_question(question):
    chunks, tier = retrieve(question)

    if not chunks:
        print("No relevant chunks found in corpus.")
        return

    answer = generate_answer(question, chunks)
    check = verify_citations(answer, chunks)

    print(f"\n[Retrieval tier: {tier}]")
    print(f"[Retrieved {len(chunks)} chunks from: {sorted(set(c['metadata']['source_doc'] for c in chunks))}]")
    print(f"\nAnswer:\n{answer}")
    print(f"\n[Citation check: {'OK' if check['verified'] else 'UNVERIFIED -> ' + str(check['unverified_citations'])}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # single question from command line
        answer_question(" ".join(sys.argv[1:]))
    else:
        # interactive loop
        print("KG-RAG QA system. Type a question (or 'exit').")
        while True:
            q = input("\n> ")
            if q.lower() in ("exit", "quit"):
                break
            answer_question(q)