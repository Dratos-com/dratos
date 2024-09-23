import { NewPostPageComponent } from "@/components/new-post-page"

type NewPostPageProps = {
  parentCommit: string
}

export default function NewPostPage({ parentCommit }: NewPostPageProps) {
  return <NewPostPageComponent parentCommit={parentCommit} />
}
